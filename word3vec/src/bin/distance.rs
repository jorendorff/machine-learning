use std::cmp::Reverse;
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use ordered_float::OrderedFloat;
use word3vec::{dot, normalize, Vectors};

/// number of closest words that will be shown
const N: usize = 40;

#[derive(Parser)]
struct Options {
    /// Contains word projections in the BINARY FORMAT.
    #[arg(value_name = "FILE")]
    file_name: PathBuf,
}

fn main() {
    let options = Options::parse();

    let vectors = Vectors::load(&options.file_name).unwrap();

    let mut line = String::new();
    'outer: loop {
        print!("Enter word or sentence (EXIT to break): ");
        let _ = std::io::stdout().flush();

        line.clear();
        match std::io::stdin().read_line(&mut line) {
            Err(err) => {
                eprintln!("error reading stdin: {err}");
                break;
            }
            Ok(0) => break,
            Ok(_) => {}
        }
        if line.trim() == "EXIT" {
            break;
        }

        let mut bi: Vec<usize> = vec![];
        for word in line.trim().split(' ') {
            let b = vectors.lookup_word(word);
            println!();
            print!("Word: {word}  Position in vocabulary: ");
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

        let size = vectors.size();
        let mut vec = vec![0.0f32; size];
        for &i in &bi {
            let row = &vectors[i];
            for (v, r) in vec.iter_mut().zip(row.iter().copied()) {
                *v += r;
            }
        }
        normalize(&mut vec);

        let mut best: Vec<(&str, f32)> = (0..vectors.num_words())
            .filter(|c| !bi.contains(c))
            .map(|c| {
                let dist = dot(&vec, &vectors[c]);
                (vectors.word(c), dist)
            })
            .collect();
        best.sort_by_key(|(_word, dist)| Reverse(OrderedFloat(*dist)));
        for (word, dist) in best.iter().take(N) {
            println!("{:>50}\t\t{:8.6}", word, dist);
        }
    }
}
