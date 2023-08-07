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
        print!("Enter three words (EXIT to break): ");
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
            println!();
            print!("Word: {word}  Position in vocabulary: ");
            match vectors.lookup_word(word) {
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

        if bi.len() != 3 {
            println!("{} words were entered.. three words are needed at the input to perform the calculation", bi.len());
            continue;
        }

        println!();
        println!("                                              Word       Cosine distance");
        println!("------------------------------------------------------------------------");

        let mut vec = vec![0.0f32; vectors.size()];
        let a = &vectors[bi[0]];
        let b = &vectors[bi[1]];
        let c = &vectors[bi[2]];
        for i in 0..vectors.size() {
            vec[i] = b[i] - a[i] + c[i];
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

    /*
    FILE *f;
    char st1[max_size];
    char bestw[N][max_size];
    char file_name[max_size], st[100][max_size];
    float dist, len, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    float *M;
    char *vocab;


      printf("\n                                              Word              Distance\n------------------------------------------------------------------------\n");
      for (a = 0; a < size; a++) vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];
      len = 0;
      for (a = 0; a < size; a++) len += vec[a] * vec[a];
      len = sqrt(len);
      for (a = 0; a < size; a++) vec[a] /= len;
      for (a = 0; a < N; a++) bestd[a] = 0;
      for (a = 0; a < N; a++) bestw[a][0] = 0;
      for (c = 0; c < words; c++) {
        if (c == bi[0]) continue;
        if (c == bi[1]) continue;
        if (c == bi[2]) continue;
        a = 0;
        for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
        for (a = 0; a < N; a++) {
          if (dist > bestd[a]) {
            for (d = N - 1; d > a; d--) {
              bestd[d] = bestd[d - 1];
              strcpy(bestw[d], bestw[d - 1]);
            }
            bestd[a] = dist;
            strcpy(bestw[a], &vocab[c * max_w]);
            break;
          }
        }
      }
      for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);

       */
}
