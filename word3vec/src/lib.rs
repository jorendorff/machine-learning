use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::ops::Index;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

pub struct Vectors {
    /// Embedding vector length (number of dimensions).
    size: usize,

    /// The vocabulary.
    vocab: Vec<String>,

    /// `embeddings[k * size..(k+1) * size]` is the vector embedding for word `k`.
    embeddings: Vec<f32>,
}

pub fn norm(v: &[f32]) -> f32 {
    v.iter().copied().map(|e| e * e).sum::<f32>().sqrt()
}

pub fn normalize(v: &mut [f32]) {
    let len = norm(v);
    for e in v {
        *e /= len;
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum()
}

impl Index<usize> for Vectors {
    type Output = [f32];

    fn index(&self, i: usize) -> &[f32] {
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
            f.read_exact(bytemuck::cast_slice_mut::<f32, u8>(row))
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
