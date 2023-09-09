use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Seek};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::{process, slice};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle, ProgressState};

// copied from word3vec/src/main.rs
const MAX_SENTENCE_LENGTH: usize = 1000;

// copied from word3vec/src/main.rs
const MAX_STRING: usize = 100;

// copied from word3vec/src/main.rs
#[allow(non_camel_case_types)]
type real = f32; // Precision of float numbers

// copied from word3vec/src/main.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VocabWord {
    cn: u64,
    point: Vec<u32>,
    word: String,
    code: Vec<u8>,
}

// copied from word3vec/src/main.rs
#[derive(Debug, Serialize, Deserialize)]
struct Model {
    size: usize,
    sample: real,
    window: usize,

    vocab: Vec<VocabWord>,
    embeddings: Vec<real>,
    weights: Vec<real>,
}

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

fn sigmoid(x: real) -> real {
    1.0 / (1.0 + (-x).exp())
}

fn dot(a: &[real], b: &[real]) -> real {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum()
}

impl Model {
    fn load(filename: &Path) -> Result<Self> {
        let f = BufReader::new(
            File::open(filename)
                .with_context(|| format!("failed to open model file {filename:?}"))?,
        );
        let model: Model = bincode::deserialize_from(f)
            .with_context(|| format!("failed to load model from file {filename:?}"))?;

        let nwords = model.vocab.len();
        let size = model.size;
        anyhow::ensure!(nwords >= 2, "invalid model: no words");
        anyhow::ensure!(
            model.vocab[0].word == "</s>",
            "invalid model: end-of-sentence marker missing"
        );

        anyhow::ensure!(
            model.embeddings.len() == nwords * size,
            "invalid model: length of embeddings array {} must be number of words in vocab {nwords} times embedding size {size}",
            model.embeddings.len(),
        );
        anyhow::ensure!(
            model.weights.len() == (nwords - 1) * size,
            "invalid model: length of weights array {} must be number of tree nodes {} times embedding size {size}",
            model.weights.len(),
            nwords - 1,
        );
        for vw in &model.vocab {
            anyhow::ensure!(vw.point.len() == vw.code.len());
            for i in 0..vw.point.len() - 1 {
                anyhow::ensure!(
                    vw.point[i] > vw.point[i + 1],
                    "paths must be strictly decreasing"
                );
            }
        }

        Ok(model)
    }

    /// Estimate P(b|a), the probability that a word in the context of `a` is `b`.
    /// (Since we compute the probability for every b, it's cheaper to do it in bulk;
    /// see `Predictor` below)
    #[allow(dead_code)]
    fn predict(&self, a: usize, b: usize) -> real {
        let size = self.size;
        let va = &self.embeddings[a * size..][..size];
        let point = &self.vocab[b].point;
        let code = &self.vocab[b].code;
        assert_eq!(point.len(), code.len());
        let mut p: real = 1.0;
        for (&node, &dir) in point.iter().zip(code.iter()) {
            let left = sigmoid(dot(va, &self.weights[node as usize * size..][..size]));
            p *= if dir == 0 { left } else { 1.0 - left };
        }
        p
    }

    // Create binary Huffman tree using the word counts.
    // Frequent words will have short unique binary codes.
    //
    // copied from word3vec/src/main.rs
    #[cfg(test)]
    #[allow(clippy::needless_range_loop)]
    fn create_binary_tree(&mut self) {
        let vocab_size = self.vocab.len();
        let mut count = vec![0u64; vocab_size * 2 + 1];
        let mut binary = vec![0u8; vocab_size * 2 + 1]; // which child a node is of its parent (0 or 1)
        let mut parent_node = vec![0usize; vocab_size * 2 + 1];

        for a in 0..vocab_size {
            count[a] = self.vocab[a].cn;
        }
        for a in vocab_size..(vocab_size * 2) {
            count[a] = 1_000_000_000_000_000;
        }

        let mut pos1 = vocab_size;
        let mut pos2 = vocab_size;
        // Following algorithm constructs the Huffman tree by adding one node at a time
        for a in 0..(vocab_size - 1) {
            // First, find two smallest nodes 'min1, min2'
            let min1i;
            if pos1 > 0 && count[pos1 - 1] < count[pos2] {
                pos1 -= 1;
                min1i = pos1;
            } else {
                min1i = pos2;
                pos2 += 1;
            }

            let min2i;
            if pos1 > 0 && count[pos1 - 1] < count[pos2] {
                pos1 -= 1;
                min2i = pos1;
            } else {
                min2i = pos2;
                pos2 += 1;
            }

            count[vocab_size + a] = count[min1i] + count[min2i];
            parent_node[min1i] = vocab_size + a;
            parent_node[min2i] = vocab_size + a;
            binary[min2i] = 1;
        }

        // Now assign binary code to each vocabulary word
        for a in 0..vocab_size {
            let mut code: Vec<u8> = vec![];
            let mut point: Vec<u32> = vec![];
            let mut b = a;
            loop {
                if !code.is_empty() {
                    point.push((b - vocab_size) as u32);
                }
                code.push(binary[b]);
                b = parent_node[b];
                if b == vocab_size * 2 - 2 {
                    break;
                }
            }
            code.reverse();
            self.vocab[a].code = code;
            point.push((vocab_size - 2) as u32);
            point.reverse();
            self.vocab[a].point = point;
        }
    }
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
            let last_step = vw.point.len() - 1;
            assert_eq!(vw.point[0] as usize, root);
            for i in 0..last_step {
                let parent = vw.point[i] as usize;
                let which_child = vw.code[i];
                let child = vw.point[i + 1] as usize;
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
            let last_step = vw.point.len() - 1;
            let parent = &self.nodes[vw.point[last_step] as usize];
            *out = parent.to_here
                * if vw.code[last_step] == 0 {
                    parent.left
                } else {
                    1.0 - parent.left
                };
        }

        &self.prob
    }
}

// copied from word3vec/src/main.rs
struct Rng(u64);

// copied from word3vec/src/main.rs
impl Rng {
    fn rand_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(25214903917).wrapping_add(11);
        self.0
    }

    /// Get a uniformly distributed random number in `0.0 .. 1.0`.
    fn rand_real(&mut self) -> real {
        (self.rand_u64() & 0xFFFF) as real / 65536.0
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

// copied from word3vec/src/main.rs
fn read_byte(fin: &mut BufReader<File>) -> Option<Result<u8, io::Error>> {
    let mut byte = 0;
    loop {
        return match fin.read(slice::from_mut(&mut byte)) {
            Ok(0) => None,
            Ok(..) => Some(Ok(byte)),
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => Some(Err(e)),
        };
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// copied from word3vec/src/main.rs
fn read_word(fin: &mut BufReader<File>) -> Result<Option<String>> {
    let mut word = vec![];
    loop {
        let b = match read_byte(fin) {
            None => return Ok(None),
            Some(result) => result.context("error reading a word")?,
        };

        if b == b'\r' {
            continue;
        }
        if b == b' ' || b == b'\t' || b == b'\n' {
            if !word.is_empty() {
                // Note: The original C code puts the whitespace character back.
                return Ok(Some(String::from_utf8_lossy(&word).to_string()));
            }
            if b == b'\n' {
                return Ok(Some("</s>".to_string()));
            } else {
                continue;
            }
        }
        if word.len() < MAX_STRING - 2 {
            word.push(b); // Truncate too long words
        }
    }
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
            train_words: model.vocab.iter().map(|vw| vw.cn).sum(),
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

    // partly copied from word3vec/src/main.rs
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
                let f = self.model.vocab[word].cn as real;
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
    pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{wide_bar:.cyan/.blue}] {bytes}/{total_bytes}")
        .unwrap());
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
        ProgressStyle::with_template("[{elapsed_precise}] [{wide_bar:.cyan/.blue}] {pos}/{len} {latest_word}")
            .unwrap()
            .with_key("latest_word", move |_state: &ProgressState, w: &mut dyn std::fmt::Write| {
                let guard = pb_latest_word.lock().unwrap();
                write!(w, "{}", &guard as &str).unwrap();
            })
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
            point: vec![],
            word,
            code: vec![],
        }
    }

    fn random_vec(rng: &mut Rng, size: usize) -> Vec<real> {
        (0..size).map(|_| 2.0 * rng.rand_real() - 1.0).collect()
    }

    fn random_model(rng: &mut Rng) -> Model {
        let size = 5;

        let nwords = 5;
        let mut vocab: Vec<VocabWord> = (0..nwords).map(|_| random_word(rng)).collect();
        vocab.sort_by_key(|vw| Reverse(vw.cn));
        vocab[0].cn = 0;
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
