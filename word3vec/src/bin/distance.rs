use anyhow::{anyhow, Result, Context};
use std::path::{Path, PathBuf};
use std::io::{BufReader, BufRead, Read, Write};
use std::fs::File;
use clap::Parser;
use ordered_float::OrderedFloat;

/// max length of strings
const MAX_SIZE: usize = 2000;

/// number of closest words that will be shown
const N: usize = 40;

#[derive(Parser)]
struct Options {
    /// Contains word projections in the BINARY FORMAT.
    #[arg(value_name = "FILE")]
    file_name: PathBuf,
}

struct Vectors {
    words: usize,
    size: usize,
    vocab: Vec<String>,
    m: Vec<f32>,
}

fn norm(v: &[f32]) -> f32 {
    v.iter().copied().map(|e| e * e).sum::<f32>().sqrt()
}

fn normalize(v: &mut [f32]) {
    let len = norm(v);
    for e in v {
        *e /= len;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum()
}

fn load(file_name: &Path) -> Result<Vectors> {
    let mut f = BufReader::new(File::open(file_name).context("error opening input file")?);
    let mut line = String::new();
    f.read_line(&mut line).context("error reading input file")?;
    let mut fields = line.split_whitespace();
    let words: usize = fields.next().ok_or_else(|| anyhow!("invalid input file"))?.parse().context("invalid input file")?;
    let size: usize = fields.next().ok_or_else(|| anyhow!("invalid input file"))?.parse().context("invalid input file")?;

    let mut vocab: Vec<String> = vec![];
    let mut m = vec![0.0; words * size];
    for b in 0..words {
        let mut vocab_word = Vec::<u8>::with_capacity(MAX_SIZE);
        let count = f.read_until(b' ', &mut vocab_word).context("error reading input file")?;
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

    Ok(Vectors { words, size, vocab, m })
}

fn main() {
    let options = Options::parse();

    let Vectors {words, size, vocab, m} = load(&options.file_name).unwrap();

    'outer: loop {
        print!("Enter word or sentence (EXIT to break): ");
        let _ = std::io::stdout().flush();

        let mut st1 = String::with_capacity(MAX_SIZE);
        match std::io::stdin().read_line(&mut st1) {
            Err(err) => {
                eprintln!("error reading stdin: {err}");
                break;
            }
            Ok(0) => break,
            Ok(_) => {}
        }
        if st1 == "EXIT" {
            break;
        }

        let mut bi: Vec<usize> = vec![];
        for sta in st1.split(' ') {
            let b = vocab.iter().position(|v| v == sta);
            println!();
            println!("Word: {sta}  Position in vocabulary: ");
            match b {
                None => {
                    println!("None");
                    println!("Out of dictionary word!");
                    continue 'outer;
                }
                Some(i) => {
                    println!("{i}");
                    bi.push(i);
                }
            }
        }

        println!();
        println!("                                              Word       Cosine distance");
        println!("------------------------------------------------------------------------");

        let mut vec = vec![0.0f32; size];
        for &i in &bi {
            let row = &m[i * size..][..size];
            for (v, r) in vec.iter_mut().zip(row.iter().copied()) {
                *v += r;
            }
        }
        normalize(&mut vec);

        let mut best: Vec<(&str, f32)> = (0..words)
            .filter(|c| !bi.contains(c))
            .map(|c| {
                let dist = dot(&vec, &m[c * size..][..size]);
                (vocab[c].as_str(), dist)
            })
            .collect();
        best.sort_by_key(|(_word, dist)| OrderedFloat(*dist));
        for (word, dist) in best.iter().take(N) {
            println!("{:50}\t\t{}", word, dist);
        }
    }
}
