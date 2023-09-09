use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{process};

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};

use word3vec::{
    dot, read_word, real, sigmoid, Model, Rng, MAX_SENTENCE_LENGTH,
};

#[derive(Parser)]
#[command(about = "evaluate model predicted probabilities against observed joint frequences", long_about = None)]
struct Options {
    #[arg(long = "train", value_name = "FILE")]
    train_file: PathBuf,

    /// embeddings and weights in bincode format
    #[arg(long = "model", value_name = "FILE")]
    model_file: PathBuf,

    /// how much of the vocabulary to use in computing the score (useful range: 0.001 to 1.0)
    #[arg(long, default_value_t = 0.01)]
    sample_rate: real,
}

#[derive(Debug, Clone)]
struct Node {
    parent: usize,

    /// 0 if we are left child of parent, 1 if right child. (unused for the root)
    which_child: u8,

    /// Given `a`, probability of reaching this node (parent node's `to_here` times either `left`
    /// or `1 - left`).
    to_here: real,

    /// Given `a` and given we reach this node, predicted probability of branching left here.
    left: real,
}

struct Predictor<'a> {
    model: &'a Model,

    /// Buffer for intermediate results.
    nodes: Vec<Node>,

    /// Output. Probability of each word.
    prob: Vec<real>,
}

impl<'a> Predictor<'a> {
    fn new(model: &'a Model) -> Self {
        // Reverse-engineer the tree from the VocabWords.
        let num_nodes = model.vocab.len() - 1;
        let root = num_nodes - 1;
        let mut nodes = vec![
            Node {
                parent: usize::MAX,
                which_child: u8::MAX,
                to_here: 0.0,
                left: 0.0
            };
            num_nodes
        ];
        for vw in &model.vocab {
            let last_step = vw.decision_indexes.len() - 1;
            assert_eq!(vw.decision_indexes[0] as usize, root);
            for i in 0..last_step {
                let parent = vw.decision_indexes[i] as usize;
                let which_child = vw.decision_path[i];
                let child = vw.decision_indexes[i + 1] as usize;
                assert!(nodes[child].parent == parent || nodes[child].parent == usize::MAX);
                nodes[child].parent = parent;
                nodes[child].which_child = which_child;
            }
        }
        // Check that above algorithm actually hit all nodes
        for (i, node) in nodes.iter().enumerate() {
            if i == root {
                assert_eq!(node.parent, usize::MAX);
                assert_eq!(node.which_child, u8::MAX);
            } else {
                assert_ne!(node.parent, usize::MAX);
                assert_ne!(node.which_child, u8::MAX);
            }
        }

        Predictor {
            model,
            nodes,
            prob: vec![0.0; model.vocab.len()],
        }
    }

    /// Predict joint probabilities. Returns an array of `predictions` such that `predictions[b] =
    /// P(b|a)`, that is, the probability, given a use of the word `a`, that a nearby word, chosen
    /// from `a`'s context with the same distribution as during training, is `b`.
    ///
    /// The `predictions` sum to 1 (except for floating-point rounding).
    fn predict(&mut self, a: usize) -> &[real] {
        let size = self.model.size;
        let va = &self.model.embeddings[a * size..][..size];

        // Predict probabilities for each node.
        let nodes = &mut self.nodes;
        let num_nodes = nodes.len();
        for i in (0..num_nodes).rev() {
            let me = &self.nodes[i];
            let prob = if i == num_nodes - 1 {
                1.0
            } else {
                let parent = &self.nodes[me.parent];
                parent.to_here
                    * if me.which_child == 0 {
                        parent.left
                    } else {
                        1.0 - parent.left
                    }
            };

            let me = &mut self.nodes[i];
            me.to_here = prob;
            me.left = sigmoid(dot(va, &self.model.weights[i * size..][..size]));
        }

        // Predict probabilities for each word.
        for (out, vw) in self.prob.iter_mut().zip(self.model.vocab.iter()) {
            let last_step = vw.decision_indexes.len() - 1;
            let parent = &self.nodes[vw.decision_indexes[last_step] as usize];
            *out = parent.to_here
                * if vw.decision_path[last_step] == 0 {
                    parent.left
                } else {
                    1.0 - parent.left
                };
        }

        &self.prob
    }
}

struct SentenceReader<'a> {
    model: &'a Model,
    file: BufReader<File>,
    file_len: u64,
    vocab_hash: HashMap<String, usize>,
    train_words: u64,
    rng: Rng,
}

impl<'a> SentenceReader<'a> {
    fn open(options: &Options, model: &'a Model) -> Result<Self> {
        let filename = &options.train_file;
        let file = File::open(filename)
            .with_context(|| format!("failed to open training file {filename:?}"))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("failed to get length of training file {filename:?}"))?
            .len();
        let file = BufReader::new(file);

        Ok(SentenceReader {
            model,
            file,
            file_len,
            vocab_hash: model
                .vocab
                .iter()
                .enumerate()
                .map(|(i, v)| (v.word.clone(), i))
                .collect(),
            train_words: model.vocab.iter().map(|vw| vw.count).sum(),
            rng: Rng(1),
        })
    }

    fn stream_position(&mut self) -> Result<u64> {
        Ok(self.file.stream_position()?)
    }

    /// Reads a word and returns its index in the vocabulary
    ///
    /// returns `Ok(None)` at end of file, `Ok(Some(None))` if a word was read
    /// but it's unrecognized.
    fn read_word_index(&mut self) -> Result<Option<Option<usize>>> {
        Ok(read_word(&mut self.file)?.map(|word| self.search_vocab(&word)))
    }

    /// Returns position of a word in the vocabulary; if the word is not found, returns None.
    fn search_vocab(&self, word: &str) -> Option<usize> {
        self.vocab_hash.get(word).copied()
    }
}

impl<'a> Iterator for SentenceReader<'a> {
    type Item = Result<Vec<usize>>;

    // TODO: This duplicates code in word3vec/src/bin/word3vec.rs.
    fn next(&mut self) -> Option<Result<Vec<usize>>> {
        let mut sen = vec![];
        loop {
            let word = match self.read_word_index() {
                Err(err) => {
                    return Some(Err(err.context("error reading a word from training data")))
                }
                Ok(None) => return None,
                Ok(Some(None)) => continue,
                Ok(Some(Some(0))) => break, // end of sentence
                Ok(Some(Some(i))) => i,
            };

            // The subsampling randomly discards frequent words while keeping the ranking same
            let sample = self.model.sample;
            if sample > 0.0 {
                let f = self.model.vocab[word].count as real;
                let k = sample * self.train_words as real;
                let ran = ((f / k).sqrt() + 1.0) * k / f;
                if ran < self.rng.rand_real() {
                    continue;
                }
            }
            sen.push(word);
            if sen.len() >= MAX_SENTENCE_LENGTH {
                break;
            }
        }
        Some(Ok(sen))
    }
}

fn window_around(i: usize, radius: usize, len: usize) -> Range<usize> {
    let start = i.saturating_sub(radius);
    let stop = (i + radius + 1).min(len);
    start..stop
}

fn evaluate_model(options: Options) -> Result<()> {
    let model = Model::load(&options.model_file)?;

    let mut rng = Rng(1);
    let mut is_selected: Vec<bool> = (0..model.vocab.len())
        .map(|_| rng.rand_real() < options.sample_rate)
        .collect();
    is_selected[0] = false; // never care about the end-of-sentence marker
    let num_words_selected = is_selected.iter().filter(|&&x| x).count();
    anyhow::ensure!(
        num_words_selected > 0,
        "sample rate {} is too low: no words selected",
        options.sample_rate
    );

    // Compute observed joint frequencies.
    //
    // The goal is to produce `freq`, a sparse square matrix such that `freq[a][b]` = the frequency
    // of b among a's neighbors. Each row freq[a] sums to neighbors[a], up to f32 rounding.
    //
    // Each occurrence of a word `b` near `a` is weighted in proportion to how often word2vec would
    // sample that word pair -- it only samples pairs at distance `option.window` or less, and
    // samples them more the closer together they are.
    let mut freq: Vec<HashMap<u32, f32>> = vec![HashMap::new(); model.vocab.len()];
    // For each word `a`, `neighbors[a]` is the total weight of a's neighbors in the entire data set.
    let mut neighbors: Vec<f32> = vec![0.0; model.vocab.len()];
    let radius = model.window;
    let fraction = 1.0 / (radius * (radius + 1)) as real;

    println!("Computing observed joint frequencies...");
    let mut sentence_reader = SentenceReader::open(&options, &model)?;
    let pb = ProgressBar::new(sentence_reader.file_len);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/.blue}] {bytes}/{total_bytes}",
        )
        .unwrap(),
    );
    while let Some(sentence) = sentence_reader.next() {
        let sentence = sentence?;
        for (i, &a) in sentence.iter().enumerate() {
            if is_selected[a] {
                let window = window_around(i, radius, sentence.len());
                for j in window {
                    if i != j {
                        let b = sentence[j];
                        let weight = (radius + 1 - i.abs_diff(j)) as f32 * fraction;
                        *freq[a].entry(b as u32).or_insert(0.0) += weight;
                        neighbors[a] += weight;
                    }
                }
            }
        }
        pb.set_position(sentence_reader.stream_position()?);
    }
    pb.finish_with_message("done");

    // Compare them to predicted conditional probabilities.
    // For each row, we have two probability distributions: `freq[a]` and `model.predict(a)`.
    // Similarity is computed using the dot product.
    let mut sum_sim = 0.0;
    let mut vfa = vec![0.0; model.vocab.len()];
    let mut predictor = Predictor::new(&model);
    let latest_word = Arc::new(Mutex::new(String::new()));
    let pb = ProgressBar::new(model.vocab.len() as u64);
    let pb_latest_word = Arc::clone(&latest_word);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/.blue}] {pos}/{len} {latest_word}",
        )
        .unwrap()
        .with_key(
            "latest_word",
            move |_state: &ProgressState, w: &mut dyn std::fmt::Write| {
                let guard = pb_latest_word.lock().unwrap();
                write!(w, "{}", &guard as &str).unwrap();
            },
        ),
    );
    for (a, fa) in freq.into_iter().enumerate() {
        if is_selected[a] {
            vfa.fill(0.0);
            let mut sum_f2 = 0.0;
            for (b, f) in fa {
                sum_f2 += f * f;
                vfa[b as usize] = f;
            }

            let vpa = predictor.predict(a);
            let sum_p2 = dot(vpa, vpa);

            let similarity = dot(&vfa, vpa) / (sum_f2.sqrt() * sum_p2.sqrt());
            {
                let mut guard = latest_word.lock().unwrap();
                *guard = format!("{:>15} - {similarity:6.4?}", model.vocab[a].word);
            }
            sum_sim += similarity;
        }
        pb.set_position(a as u64);
    }
    pb.finish_with_message("done");

    // Scale q to a percentage of the maximum possible score, which would be obtained if every row
    // matched the observed frequencies exactly, producing a dot product of 1 for each row.
    let grade = 100.0 * sum_sim / num_words_selected as f32;
    println!("Model quality: {grade:?}%");

    Ok(())
}

fn main() {
    let options = Options::parse();

    if let Err(err) = evaluate_model(options) {
        eprintln!("{err:#}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use super::*;

    fn random_word(rng: &mut Rng) -> VocabWord {
        let word = (0..8)
            .map(|_| char::from(b'a' + (rng.rand_u64() % 26) as u8))
            .collect::<String>();
        VocabWord {
            cn: (10.0 * rng.rand_real()).exp().max(5.0).floor() as u64,
            decision_indexes: vec![],
            word,
            decision_path: vec![],
        }
    }

    fn random_vec(rng: &mut Rng, size: usize) -> Vec<real> {
        (0..size).map(|_| 2.0 * rng.rand_real() - 1.0).collect()
    }

    fn random_model(rng: &mut Rng) -> Model {
        let size = 5;

        let nwords = 5;
        let mut vocab: Vec<VocabWord> = (0..nwords).map(|_| random_word(rng)).collect();
        vocab.sort_by_key(|vw| Reverse(vw.count));
        vocab[0].count = 0;
        vocab[0].word = "</s>".to_string();

        let mut model = Model {
            size,
            sample: 0.0,
            window: 4,
            vocab,
            embeddings: random_vec(rng, nwords * size),
            weights: random_vec(rng, (nwords - 1) * size),
        };
        model.create_binary_tree();
        model
    }

    #[track_caller]
    fn assert_near(left: real, right: real) {
        assert!(
            (
                // either both are effectively 0;
                left.abs() < 1e-10 && right.abs() < 1e-10
            ) || (
                // or both have the same sign and differ by less than 0.1%
                (left < 0.0) == (right < 0.0)
                    && (left - right).abs() < 0.001 * left.abs().min(right.abs())
            ),
            "assertion failed: expected approximately equal values, got {left:?} and {right:?}"
        );
    }

    // We have two ways of computing predictions; make sure they give the same answers.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_prediction_methods_agree() {
        let mut rng = Rng(1);
        let model = random_model(&mut rng);
        println!("model: {model:#?}");
        let mut predictor = Predictor::new(&model);
        for a in 1..model.vocab.len() {
            let vpa = predictor.predict(a);
            let vpa_alt: Vec<real> = (0..model.vocab.len())
                .map(|b| model.predict(a, b))
                .collect();
            println!("{a}: {vpa:?}");
            println!("{a}: {vpa_alt:?}");
            for b in 1..model.vocab.len() {
                assert_near(vpa[b], model.predict(a, b));
            }
        }
    }
}
