use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, BufReader, ErrorKind};
use std::path::{Path, PathBuf};
use std::{process, slice};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

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
#[derive(Serialize, Deserialize)]
struct Model {
    vocab: Vec<VocabWord>,
    embeddings: Vec<f32>,
    weights: Vec<f32>,
}

#[derive(Parser)]
#[command(about = "evaluate model predicted probabilities against observed joint frequences", long_about = None)]
struct Options {
    #[arg(long = "train", value_name = "FILE")]
    train_file: PathBuf,

    ///  embeddings and weights in bincode format
    #[arg(long = "model", value_name = "FILE")]
    model_file: PathBuf,

    /// Set threshold for occurrence of words. Those that appear with higher
    /// frequency in the training data will be randomly down-sampled; default
    /// is 1e-3, useful range is (0, 1e-5)
    #[arg(long, default_value_t = 1e-3)]
    sample: real,
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
        bincode::deserialize_from(f)
            .with_context(|| format!("failed to load model from file {filename:?}"))
    }

    fn size(&self) -> usize {
        self.embeddings.len() / self.vocab.len()
    }

    /// Estimate P(b|a), the probability that a word in the context of `a` is `b`.
    /// (If we want to compute the probability for every b, it'd cheaper to do it in bulk.)
    #[allow(dead_code)]
    fn predict(&self, a: usize, b: usize) -> real {
        let size = self.size();
        let va = &self.embeddings[a * size..][..size];
        let point = &self.vocab[b].point;
        let code = &self.vocab[b].code;
        assert_eq!(point.len(), code.len());
        let mut p: real = 1.0;
        for (&node, &dir) in point.iter().zip(code.iter()) {
            let sign = if dir == 0 { 1.0 } else { -1.0 };
            p *= sigmoid(sign * dot(va, &self.weights[node as usize * size..][..size]));
        }
        p
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
    vocab_hash: HashMap<String, usize>,
    train_words: u64,
    rng: Rng,
    sample: real,
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
        let file = BufReader::new(
            File::open(filename)
                .with_context(|| format!("failed to open raining file {filename:?}"))?,
        );

        Ok(SentenceReader {
            model,
            file,
            vocab_hash: model
                .vocab
                .iter()
                .enumerate()
                .map(|(i, v)| (v.word.clone(), i))
                .collect(),
            train_words: model.vocab.iter().map(|vw| vw.cn).sum(),
            rng: Rng(1),
            sample: options.sample,
        })
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
            let sample = self.sample;
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

fn evaluate_model(options: Options) -> Result<()> {
    let model = Model::load(&options.model_file)?;

    // Compute observed joint frequencies
    let sentence_reader = SentenceReader::open(&options, &model)?;
    for sentence in sentence_reader {
        let _sentence = sentence?;
        // TODO accumulate stats
    }

    // TODO: compute predicted conditional probabilities

    // TODO: compare and compute accuracy

    Ok(())
}

fn main() {
    let options = Options::parse();

    if let Err(err) = evaluate_model(options) {
        eprintln!("{err:#}");
        process::exit(1);
    }
}
