use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, ErrorKind};
use std::ops::Index;
use std::path::Path;
use std::slice;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

pub const MAX_STRING: usize = 100;
pub const MAX_SENTENCE_LENGTH: usize = 1000;

#[allow(non_camel_case_types)]
pub type real = f32; // Precision of float numbers

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabWord {
    pub count: u64,
    pub word: String,
    pub decision_indexes: Vec<u32>,
    pub decision_path: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub size: usize,
    pub sample: real,
    pub window: usize,
    pub vocab: Vec<VocabWord>,
    pub embeddings: Vec<real>,
    pub weights: Vec<real>,
}

pub fn norm(v: &[real]) -> real {
    v.iter().copied().map(|e| e * e).sum::<real>().sqrt()
}

pub fn normalize(v: &mut [real]) {
    let len = norm(v);
    for e in v {
        *e /= len;
    }
}

pub fn dot(a: &[real], b: &[real]) -> real {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum()
}

pub fn sigmoid(x: real) -> real {
    1.0 / (1.0 + (-x).exp())
}

pub struct Rng(pub u64);

impl Rng {
    pub fn rand_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(25214903917).wrapping_add(11);
        self.0
    }

    /// Get a uniformly distributed random number in `0.0 .. 1.0`.
    pub fn rand_real(&mut self) -> real {
        (self.rand_u64() & 0xFFFF) as real / 65536.0
    }
}

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
pub fn read_word(fin: &mut BufReader<File>) -> Result<Option<String>> {
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

pub struct Vectors {
    /// Embedding vector length (number of dimensions).
    size: usize,

    /// The vocabulary.
    vocab: Vec<String>,

    /// `embeddings[k * size..(k+1) * size]` is the vector embedding for word `k`.
    embeddings: Vec<real>,
}

impl Index<usize> for Vectors {
    type Output = [real];

    fn index(&self, i: usize) -> &[real] {
        &self.embeddings[i * self.size..][..self.size]
    }
}

impl Vectors {
    pub fn load(file_name: &Path) -> Result<Self> {
        let mut f = BufReader::new(File::open(file_name).context("error opening input file")?);
        let mut line = String::new();
        f.read_line(&mut line).context("error reading input file")?;
        let mut fields = line.split_whitespace();
        let num_words: usize = fields
            .next()
            .ok_or_else(|| anyhow!("invalid input file"))?
            .parse()
            .context("invalid input file")?;
        let size: usize = fields
            .next()
            .ok_or_else(|| anyhow!("invalid input file"))?
            .parse()
            .context("invalid input file")?;

        let mut vocab: Vec<String> = vec![];
        let mut m = vec![0.0; num_words * size];
        for b in 0..num_words {
            let mut vocab_word = Vec::<u8>::new();
            let count = f
                .read_until(b' ', &mut vocab_word)
                .context("error reading input file")?;
            if count == 0 {
                break;
            }
            if vocab_word.last() == Some(&b' ') {
                vocab_word.pop();
            }
            vocab_word.retain(|c| *c != b'\n');
            vocab.push(String::from_utf8(vocab_word).context("invalid word in input file")?);

            let row = &mut m[b * size..][..size];
            f.read_exact(bytemuck::cast_slice_mut::<real, u8>(row))
                .context("error reading input file")?;
            normalize(row);
        }

        Ok(Vectors {
            size,
            vocab,
            embeddings: m,
        })
    }

    pub fn num_words(&self) -> usize {
        self.vocab.len()
    }

    /// Returns the vector size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the index for a word as string. Exact match only, case-sensitive.
    pub fn lookup_word(&self, word: &str) -> Option<usize> {
        self.vocab.iter().position(|v| v == word)
    }

    /// Get the word for a word-index. Panics if `word` is out of range.
    pub fn word(&self, word: usize) -> &str {
        &self.vocab[word]
    }
}

impl Model {
    pub fn load(filename: &Path) -> Result<Self> {
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
            anyhow::ensure!(vw.decision_indexes.len() == vw.decision_path.len());
            for i in 0..vw.decision_indexes.len() - 1 {
                anyhow::ensure!(
                    vw.decision_indexes[i] > vw.decision_indexes[i + 1],
                    "paths must be strictly decreasing"
                );
            }
        }

        Ok(model)
    }

    /// Estimate P(b|a), the probability that a word in the context of `a` is `b`.
    /// (Since we compute the probability for every b, it's cheaper to do it in bulk;
    /// see `Predictor` below)
    pub fn predict(&self, a: usize, b: usize) -> real {
        let size = self.size;
        let va = &self.embeddings[a * size..][..size];
        let point = &self.vocab[b].decision_indexes;
        let code = &self.vocab[b].decision_path;
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
    #[allow(clippy::needless_range_loop)]
    pub fn create_binary_tree(&mut self) {
        let vocab_size = self.vocab.len();
        let mut count = vec![0u64; vocab_size * 2 + 1];
        let mut binary = vec![0u8; vocab_size * 2 + 1]; // which child a node is of its parent (0 or 1)
        let mut parent_node = vec![0usize; vocab_size * 2 + 1];

        for a in 0..vocab_size {
            count[a] = self.vocab[a].count;
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
            let mut path: Vec<u8> = vec![];
            let mut indexes: Vec<u32> = vec![];
            let mut b = a;
            loop {
                if !path.is_empty() {
                    indexes.push((b - vocab_size) as u32);
                }
                path.push(binary[b]);
                b = parent_node[b];
                if b == vocab_size * 2 - 2 {
                    break;
                }
            }
            path.reverse();
            self.vocab[a].decision_path = path;
            indexes.push((vocab_size - 2) as u32);
            indexes.reverse();
            self.vocab[a].decision_indexes = indexes;
        }
    }
}
