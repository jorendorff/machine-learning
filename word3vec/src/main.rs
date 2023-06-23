#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use std::{iter, process, slice, thread};

use aligned_box::AlignedBox;
use anyhow::{Context, Result};
use clap::Parser;

const MAX_STRING: usize = 100;
const EXP_TABLE_SIZE: usize = 1000;
const MAX_EXP: real = 6.0;
const MAX_SENTENCE_LENGTH: usize = 1000;
const MAX_CODE_LENGTH: usize = 40;

const VOCAB_HASH_SIZE: usize = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

#[allow(non_camel_case_types)]
type real = f32; // Precision of float numbers

struct VocabWord {
    cn: u64,
    point: Vec<u32>,
    word: String,
    code: Vec<u8>,
}

#[derive(Parser)]
#[command(about = "WORD VECTOR estimation toolkit", long_about = None, version = "0.1d")]
struct Options {
    /// Use text data from FILE to train the model
    #[arg(long = "train", value_name = "FILE")]
    train_file: PathBuf,

    /// Use FILE to save the resulting word vectors / word clusters
    #[arg(long = "output", value_name = "FILE")]
    output_file: Option<PathBuf>,

    /// Set size of word vectors; default is 100
    #[arg(long = "size", default_value_t = 100)]
    layer1_size: usize,

    /// Set max skip length between words
    #[arg(long, default_value_t = 5)]
    window: usize,

    /// Set threshold for occurrence of words. Those that appear with higher
    /// frequency in the training data will be randomly down-sampled; default
    /// is 1e-3, useful range is (0, 1e-5)
    #[arg(long, default_value_t = 1e-3)]
    sample: real,

    /// Use Hierarchical Softmax
    #[arg(long)]
    hs: bool,

    /// Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
    #[arg(long, default_value_t = 5)]
    negative: i32,

    /// Use N threads
    #[arg(long = "threads", value_name = "N", default_value_t = 12)]
    num_threads: usize,

    /// Run more training iterations
    #[arg(long, default_value_t = 5)]
    iter: usize,

    /// Discard words that appear less than N times
    #[arg(long = "min-count", value_name = "N", default_value_t = 5)]
    min_count: u64,

    /// Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
    #[arg(long)]
    alpha: Option<real>,

    /// Output word classes rather than word vectors; if unspecified, vectors are written instead
    #[arg(long)]
    classes: Option<usize>,

    /// Set the debug mode (default = 2 = more info during training)
    #[arg(long = "debug", default_value_t = 2)]
    debug_mode: usize,

    /// Save the resulting vectors in binary mode
    #[arg(long)]
    binary: bool,

    /// The vocabulary will be saved to FILE
    #[arg(long = "save-vocab", value_name = "FILE")]
    save_vocab_file: Option<PathBuf>,

    /// The vocabulary will be read from FILE, not constructed from the training data
    #[arg(long = "read-vocab", value_name = "FILE")]
    read_vocab_file: Option<PathBuf>,

    /// Use the continuous bag of words model (otherwise, use skip-gram model)
    // Note: In the original, cbow was the default; you specified skip-grams with `-cbow 0`.
    #[arg(long)]
    cbow: bool,
}

#[derive(Default)]
#[repr(transparent)]
struct Real {
    bits: AtomicU32,
}

impl Real {
    fn get(&self) -> real {
        real::from_bits(self.bits.load(Ordering::Relaxed))
    }

    fn set(&self, value: real) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }

    fn add(&self, x: real) {
        let a = self.get();
        self.set(a + x);
    }
}

struct Word3Vec {
    options: Options,
    vocab: Vec<VocabWord>,
    min_reduce: u64,
    vocab_hash: HashMap<String, usize>,
    train_words: u64,
    word_count_actual: AtomicU64,
    file_size: u64,
    starting_alpha: real,
    /// The learned word-vectors.
    syn0: AlignedBox<[Real]>,
    syn1: AlignedBox<[Real]>,
    syn1neg: AlignedBox<[Real]>,
    exp_table: Vec<real>,
    start: Instant,
    table: Vec<usize>,
}

const TABLE_SIZE: usize = 100_000_000;

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
fn read_word(fin: &mut BufReader<File>) -> Result<Option<String>> {
    let mut word = vec![];
    loop {
        let b = match read_byte(fin) {
            None => break,
            Some(result) => result.context("error reading a word")?,
        };

        if b == b'\r' {
            continue;
        }
        if b == b' ' || b == b'\t' || b == b'\n' {
            if !word.is_empty() {
                // TODO: PUT THE DAMN THING BACK
                break;
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
    Ok(if word.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&word).to_string())
    })
}

// Read words from a file, assuming space + tab + EOL to be word boundaries
fn read_words(fin: File) -> impl Iterator<Item = Result<String, io::Error>> {
    let mut bytes = BufReader::new(fin).bytes().peekable();
    iter::from_fn(move || -> Option<Result<String, io::Error>> {
        let mut word = Vec::<u8>::new();
        while let Some(res) = bytes.peek() {
            let ch = match res {
                Err(_err) => return Some(Err(bytes.next().unwrap().unwrap_err())),
                Ok(ch) => *ch,
            };
            if ch == b'\r' {
                bytes.next();
                continue;
            }
            if ch == b' ' || ch == b'\t' || ch == b'\n' {
                if !word.is_empty() {
                    break;
                }
                bytes.next();
                if ch == b'\n' {
                    return Some(Ok("</s>".to_string()));
                } else {
                    continue;
                }
            }
            bytes.next();
            if word.len() < MAX_STRING - 1 {
                word.push(ch); // Truncate too long words
            }
        }
        if word.is_empty() {
            None
        } else {
            Some(Ok(String::from_utf8_lossy(&word).to_string()))
        }
    })
}

impl Word3Vec {
    fn new(options: Options) -> Self {
        let exp_table = (0..EXP_TABLE_SIZE)
            .map(|i| {
                let e = ((i as real / EXP_TABLE_SIZE as real * 2.0 - 1.0) * MAX_EXP).exp(); // Precompute the exp() table
                e / (e + 1.0) // Precompute f(x) = x / (x + 1)
            })
            .collect();

        Word3Vec {
            options,
            vocab: Vec::with_capacity(1000),
            min_reduce: 1,
            vocab_hash: HashMap::new(),
            train_words: 0,
            word_count_actual: AtomicU64::new(0),
            file_size: 0,
            starting_alpha: 0.0,
            syn0: AlignedBox::slice_from_default(128, 128).unwrap(),
            syn1: AlignedBox::slice_from_default(128, 128).unwrap(),
            syn1neg: AlignedBox::slice_from_default(128, 128).unwrap(),
            exp_table,
            start: Instant::now(),
            table: Vec::with_capacity(TABLE_SIZE),
        }
    }

    fn init_unigram_table(&mut self) {
        let power: f64 = 0.75;
        let train_words_pow = self
            .vocab
            .iter()
            .map(|v| (v.cn as f64).powf(power))
            .sum::<f64>();

        let mut i = 0;
        let mut d1 = (self.vocab[i].cn as f64).powf(power) / train_words_pow;
        for a in 0..TABLE_SIZE {
            self.table.push(i);
            if (a as f64 / TABLE_SIZE as f64) > d1 {
                i += 1;
                d1 += (self.vocab[i].cn as f64).powf(power) / train_words_pow;
            }
            if i >= self.vocab.len() {
                i = self.vocab.len() - 1;
            }
        }
    }

    /// Returns position of a word in the vocabulary; if the word is not found, returns None.
    fn search_vocab(&self, word: &str) -> Option<usize> {
        self.vocab_hash.get(word).copied()
    }

    /// Reads a word and returns its index in the vocabulary
    ///
    /// returns `Ok(None)` at end of file, `Ok(Some(None))` if a word was read
    /// but it's unrecognized.
    fn read_word_index(&self, fin: &mut BufReader<File>) -> Result<Option<Option<usize>>> {
        Ok(read_word(fin)?.map(|word| self.search_vocab(&word)))
    }

    /// Adds a word to the vocabulary
    fn add_word_to_vocab(&mut self, word: String) -> usize {
        let n = self.vocab.len();
        self.vocab.push(VocabWord {
            cn: 0,
            point: Vec::new(),
            word: word.clone(),
            code: Vec::new(),
        });
        self.vocab_hash.insert(word, n);
        n
    }

    /// Sorts the vocabulary by frequency using word counts
    fn sort_vocab(&mut self) {
        // Sort the vocabulary and keep </s> at the first position
        self.vocab[1..].sort_by_key(|vw| Reverse(vw.cn));

        self.train_words = 0;
        self.vocab_hash.clear();
        // Words occuring less than min_count times will be discarded from the vocab
        let mut i = 0;
        self.vocab.retain(|vw| {
            let keep = i == 0 || vw.cn >= self.options.min_count;
            if keep {
                // Hash will be re-computed, as after the sorting it is not actual
                self.vocab_hash.insert(vw.word.clone(), i);
                self.train_words += vw.cn;
                i += 1;
            }
            keep
        });

        // Allocate memory for the binary tree construction
        for vw in &mut self.vocab {
            vw.code.reserve(MAX_CODE_LENGTH);
            vw.point.reserve(MAX_CODE_LENGTH);
        }
    }

    /// Reduces the vocabulary by removing infrequent tokens
    fn reduce_vocab(&mut self) {
        self.vocab.retain(|vw| vw.cn > self.min_reduce);

        // Hash will be re-computed, as it is not actual
        self.vocab_hash.clear();
        for (i, vw) in self.vocab.iter().enumerate() {
            self.vocab_hash.insert(vw.word.clone(), i);
        }
        self.min_reduce += 1;
    }

    // Create binary Huffman tree using the word counts.
    // Frequent words will have short unique binary codes.
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
                code.push(binary[b]);
                point.push(b as u32);
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

    fn learn_vocab_from_train_file(&mut self) -> Result<()> {
        let mut fin =
            File::open(&self.options.train_file).context("error opening training data file")?;
        fin.seek(SeekFrom::End(0))
            .context("error checking training data file size")?;
        self.file_size = fin
            .stream_position()
            .context("error checking training data file size")?;
        fin.seek(SeekFrom::Start(0))
            .context("error checking training data file size")?;

        self.vocab.clear();
        self.vocab_hash.clear();
        self.add_word_to_vocab("</s>".to_string());

        for word in read_words(fin) {
            let word = word.context("error reading training data file")?;
            self.train_words += 1;
            if self.options.debug_mode > 1 && self.train_words % 100_000 == 0 {
                print!("{}K\r", self.train_words / 1000);
                let _ = io::stdout().flush();
            }

            if let Some(&a) = self.vocab_hash.get(&word) {
                self.vocab[a].cn += 1;
            } else {
                let a = self.add_word_to_vocab(word);
                self.vocab[a].cn = 1;
            }

            if self.vocab.len() as f64 > VOCAB_HASH_SIZE as f64 * 0.7 {
                self.reduce_vocab();
            }
        }
        self.sort_vocab();
        if self.options.debug_mode > 0 {
            println!("Vocab size: {}", self.vocab.len());
            println!("Words in train file: {}", self.train_words);
        }
        Ok(())
    }

    fn save_vocab(&self, vocab_file: &Path) -> Result<()> {
        let mut fo = BufWriter::new(
            File::create(vocab_file).context("error creating vocab file for write")?,
        );
        for vw in &self.vocab {
            writeln!(fo, "{} {}", vw.word, vw.cn).context("error writing vocab file")?;
        }
        Ok(())
    }

    fn read_vocab(&mut self, vocab_file: &Path) -> Result<()> {
        let fin = BufReader::new(File::open(vocab_file).context("error opening vocabulary file")?);
        self.vocab_hash.clear();
        self.vocab.clear();

        for (line_num, line) in fin.lines().enumerate() {
            let line = line.context("error reading vocabulary file")?;
            let fields = line.split_whitespace().collect::<Vec<&str>>();
            anyhow::ensure!(
                fields.len() != 2,
                "vocabulary file syntax error on line {}",
                line_num + 1
            );
            let word = fields[0].to_string();
            let cn = fields[1].parse::<u64>().context(format!(
                "error reading vocabulary file: unrecognized frequency number format on line {}",
                line_num + 1
            ))?;

            let a = self.add_word_to_vocab(word);
            self.vocab[a].cn = cn;
        }
        self.sort_vocab();
        if self.options.debug_mode > 0 {
            println!("Vocab size: {}", self.vocab.len());
            println!("Words in train file: {}", self.train_words);
        }

        let mut fin =
            File::open(&self.options.train_file).context("error opening training file")?;
        fin.seek(SeekFrom::End(0))
            .context("error checking size of training file")?;
        self.file_size = fin
            .stream_position()
            .context("error checking size of training file")?;
        Ok(())
    }

    fn init_net(&mut self) {
        let vocab_size = self.vocab.len();
        let layer1_size = self.options.layer1_size;

        self.syn0 = AlignedBox::slice_from_default(128, vocab_size * layer1_size)
            .expect("Memory allocation failed");
        if self.options.hs {
            self.syn1 = AlignedBox::slice_from_default(128, vocab_size * layer1_size)
                .expect("Memory allocation failed");
        }
        if self.options.negative > 0 {
            self.syn1neg = AlignedBox::slice_from_default(128, vocab_size * layer1_size)
                .expect("Memory allocation failed");
        }

        let mut next_random = 1u64;
        for a in 0..vocab_size {
            for b in 0..layer1_size {
                next_random = next_random.wrapping_mul(25214903917).wrapping_add(11);
                self.syn0[a * layer1_size + b]
                    .set((((next_random & 0xFFFF) as real / 65536.0) - 0.5) / layer1_size as real);
            }
        }
        self.create_binary_tree();
    }

    fn train_model_thread(&self, id: usize) -> Result<()> {
        let window = self.options.window;

        let mut neu1: Vec<real> = vec![0.0; self.options.layer1_size];
        let mut neu1e: Vec<real> = vec![0.0; self.options.layer1_size];

        let mut fi = BufReader::new(File::open(&self.options.train_file)?);
        fi.seek(SeekFrom::Start(
            self.file_size / self.options.num_threads as u64 * id as u64,
        ))
        .context("error seeking within training file")?;

        let layer1_size = self.options.layer1_size;

        let mut next_random = id as u64;
        let mut alpha = self.starting_alpha;
        let mut local_iter = self.options.iter;
        let mut word_count: u64 = 0;
        let mut last_word_count: u64 = 0;
        let mut sen: Vec<usize> = Vec::with_capacity(MAX_SENTENCE_LENGTH + 1);
        let mut sentence_position: usize = 0;
        loop {
            if word_count - last_word_count > 10000 {
                let n = word_count - last_word_count;
                let word_count_actual = self.word_count_actual.fetch_add(n, Ordering::Relaxed) + n;
                last_word_count = word_count;

                if self.options.debug_mode > 1 {
                    print!(
                        "\rAlpha: {}  Progress: {:.2}%  Words/thread/sec: {:.2}k  ",
                        alpha,
                        word_count_actual as real
                            / (self.options.iter as u64 * self.train_words + 1) as real
                            * 100.0,
                        word_count_actual as real
                            / ((self.start.elapsed().as_secs_f64() + 1.0) as real * 1000.0),
                    );
                    let _ = io::stdout().flush();
                }
                alpha = self.starting_alpha
                    * (1.0
                        - word_count_actual as real
                            / (self.options.iter as u64 * self.train_words + 1) as real)
                        .max(0.0001);
            }

            let mut at_end_of_file = false;
            if sen.is_empty() {
                loop {
                    let word = self
                        .read_word_index(&mut fi)
                        .context("error reading a word from training data")?;
                    let word = match word {
                        None => {
                            at_end_of_file = true;
                            break;
                        }
                        Some(None) => continue,
                        Some(Some(i)) => i,
                    };
                    word_count += 1;
                    if word == 0 {
                        break;
                    }

                    // The subsampling randomly discards frequent words while keeping the ranking same
                    let sample = self.options.sample;
                    if sample > 0.0 {
                        let f = self.vocab[word].cn as real;
                        let k = sample * self.train_words as real;
                        let ran = ((f / k).sqrt() + 1.0) * k / f;
                        next_random = next_random.wrapping_mul(25214903917).wrapping_add(11);
                        if ran < (next_random & 0xFFFF) as real / 65536.0 {
                            continue;
                        }
                    }
                    sen.push(word);
                    if sen.len() >= MAX_SENTENCE_LENGTH {
                        break;
                    }
                }
                sentence_position = 0;
            }

            if at_end_of_file || word_count > self.train_words / self.options.num_threads as u64 {
                self.word_count_actual
                    .fetch_add(word_count - last_word_count, Ordering::Relaxed);
                local_iter -= 1;
                if local_iter == 0 {
                    break;
                }
                word_count = 0;
                last_word_count = 0;
                sen.clear();
                fi.seek(SeekFrom::Start(
                    self.file_size / self.options.num_threads as u64 * id as u64,
                ))
                .context("error rewinding file for next iteration")?;
                continue;
            }

            let word = sen[sentence_position];
            neu1.fill(0.0);
            neu1e.fill(0.0);
            next_random = next_random.wrapping_mul(25214903917).wrapping_add(11);
            let b = next_random as usize % self.options.window;

            if self.options.cbow {
                //train the cbow architecture
                // in -> hidden
                let mut cw = 0;
                for a in b..(window * 2 + 1 - b) {
                    if a != window {
                        if sentence_position + a < window {
                            continue;
                        }
                        let c = sentence_position + a - window;
                        if c >= sen.len() {
                            continue;
                        }
                        let last_word = sen[c];

                        for c in 0..layer1_size {
                            neu1[c] += self.syn0[c + last_word * layer1_size].get();
                        }
                        cw += 1;
                    }
                }

                if cw > 0 {
                    for c in 0..layer1_size {
                        neu1[c] /= cw as real;
                        if self.options.hs {
                            let vw = &self.vocab[word];
                            for d in 0..vw.code.len() {
                                let l2 = vw.point[d] as usize * layer1_size;
                                // Propagate hidden -> output
                                let f = (0..layer1_size).map(|c| neu1[c] * self.syn1[c + l2].get()).sum::<real>();
                                if f <= -MAX_EXP || f >= MAX_EXP {
                                    continue;
                                }
                                let f = self.exp_table[((f + MAX_EXP) * (EXP_TABLE_SIZE as real / MAX_EXP / 2.0)) as usize];

                                // 'g' is the gradient multiplied by the learning rate
                                let g = ((1 - vw.code[d]) as real - f) * alpha;
                                // Propagate errors output -> hidden
                                for c in 0..layer1_size {
                                    neu1e[c] += g * self.syn1[c + l2].get();
                                }
                                // Learn weights hidden -> output
                                for c in 0..layer1_size {
                                    self.syn1[c + l2].add(g * neu1[c]);
                                }
                            }
                        }
                        // NEGATIVE SAMPLING
                        if self.options.negative > 0 {
                            for d in 0..(self.options.negative  + 1) {
                                let mut target;
                                let label;
                                if d == 0 {
                                    target = word;
                                    label = 1;
                                } else {
                                    next_random = next_random.wrapping_mul(25214903917).wrapping_add(11);
                                    target = self.table[(next_random >> 16) as usize % TABLE_SIZE];
                                    if target == 0 {
                                        target = next_random as usize % (self.vocab.len() - 1) + 1;
                                    }
                                    if target == word { continue; }
                                    label = 0;
                                }

                                let l2 = target * layer1_size;
                                let f = (0..layer1_size).map(|c| neu1[c] * self.syn1neg[c + l2].get()).sum::<real>();
                                let yh = if f > MAX_EXP {
                                     1.0
                                } else if f < -MAX_EXP {
                                     0.0
                                } else {
                                     self.exp_table[((f + MAX_EXP) * (EXP_TABLE_SIZE as real / MAX_EXP / 2.0)) as usize]
                                };
                                let g = (label as real - yh) * alpha;

                                for c in 0..layer1_size { neu1e[c] += g * self.syn1neg[c + l2].get(); }
                                for c in 0..layer1_size { self.syn1neg[c + l2].add(g * neu1[c]); }
                            }
                        }

                        // hidden -> in
                        for a in b..(window * 2 + 1 - b) {
                            if a != window {
                                if sentence_position + a < window {
                                    continue;
                                }
                                let c = sentence_position + a - window;
                                if c >= sen.len() {
                                    continue;
                                }
                                let last_word = sen[c];

                                for c in 0..layer1_size {
                                    self.syn0[c + last_word * layer1_size].add(neu1e[c]);
                                }
                            }
                        }
                    }
                }
            } else {
                //train skip-gram
                for a in b..(window * 2 + 1 - b) {
                    if a != window {
                        if sentence_position + a < window {
                            continue;
                        }
                        let c = sentence_position + a - window;
                        if c >= sen.len() {
                            continue;
                        }
                        let last_word = sen[c];
                        let l1 = last_word * layer1_size;
                        neu1e.fill(0.0);
                        // HIERARCHICAL SOFTMAX
                        if self.options.hs {
                            for d in 0..self.vocab[word].code.len() {
                                // Propagate hidden -> output
                                let l2 = self.vocab[word].point[d] as usize * layer1_size;
                                let f = (0..layer1_size)
                                    .map(|c| self.syn0[l1 + c].get() * self.syn1[l2 + c].get())
                                    .sum::<real>();
                                if f <= -MAX_EXP {
                                    continue;
                                } else if f >= MAX_EXP {
                                    continue;
                                }
                                let f = self.exp_table[((f + MAX_EXP)
                                    * (EXP_TABLE_SIZE as real / MAX_EXP / 2.0))
                                    as usize];
                                // 'g' is the gradient multiplied by the learning rate
                                let g = (1.0 - self.vocab[word].code[d] as real - f) * alpha;
                                // Propagate errors output -> hidden
                                for c in 0..layer1_size {
                                    neu1e[c] += g * self.syn1[c + l2].get();
                                }
                                // Learn weights hidden -> output
                                for c in 0..layer1_size {
                                    self.syn1[c + l2].add(g * self.syn0[c + l1].get());
                                }
                            }
                        }

                        // NEGATIVE SAMPLING
                        if self.options.negative > 0 {
                            for d in 0..(self.options.negative + 1) {
                                let mut target;
                                let label;
                                if d == 0 {
                                    target = word;
                                    label = 1;
                                } else {
                                    next_random = next_random.wrapping_mul(25214903917).wrapping_add(11);
                                    target = self.table[(next_random >> 16) as usize % TABLE_SIZE];
                                    if target == 0 {
                                        target = next_random as usize % (self.vocab.len() - 1) + 1;
                                    }
                                    if target == word {
                                        continue;
                                    }
                                    label = 0;
                                }
                                let l2 = target * layer1_size;
                                let f = (0..layer1_size)
                                    .map(|c| self.syn0[c + l1].get() * self.syn1neg[c + l2].get())
                                    .sum::<real>();
                                let yh = if f > MAX_EXP {
                                    1.0
                                } else if f < -MAX_EXP {
                                    0.0
                                } else {
                                    self.exp_table[((f + MAX_EXP)
                                        * (EXP_TABLE_SIZE as real / MAX_EXP / 2.0))
                                        as usize]
                                };
                                let g = (label as real - yh) * alpha;

                                for c in 0..layer1_size {
                                    neu1e[c] += g * self.syn1neg[c + l2].get();
                                }
                                for c in 0..layer1_size {
                                    self.syn1neg[c + l2].add(g * self.syn0[c + l1].get());
                                }
                            }
                        }

                        // Learn weights input -> hidden
                        for c in 0..layer1_size {
                            self.syn0[c + l1].add(neu1e[c]);
                        }
                    }
                }
            }
            sentence_position += 1;
            if sentence_position >= sen.len() {
                sen.clear();
            }
        }

        Ok(())
    }

    fn train_model(&mut self) -> Result<()> {
        println!("Starting training using file {:?}", self.options.train_file);

        self.starting_alpha =
            self.options
                .alpha
                .unwrap_or(if self.options.cbow { 0.05 } else { 0.025 });
        if let Some(f) = self.options.read_vocab_file.clone() {
            self.read_vocab(&f)?;
        } else {
            self.learn_vocab_from_train_file()?;
        }
        if let Some(f) = &self.options.save_vocab_file {
            self.save_vocab(f)?;
        }
        let output_file = match self.options.output_file.clone() {
            Some(f) => f,
            None => return Ok(()),
        };
        self.init_net();
        if self.options.negative > 0 {
            self.init_unigram_table();
        }
        self.start = Instant::now();
        thread::scope(|s| {
            let this: &Word3Vec = self;
            let threads = (0..this.options.num_threads)
                .map(|a| s.spawn(move || this.train_model_thread(a)))
                .collect::<Vec<_>>();
            for thread in threads {
                if let Err(err) = thread.join().unwrap() {
                    eprintln!("Error in worker thread: {err:#}");
                }
            }
        });

        let mut fo =
            BufWriter::new(File::create(output_file).context("error creating output file")?);
        let vocab_size = self.vocab.len();
        let layer1_size = self.options.layer1_size;
        match self.options.classes {
            None => {
                // Save the word vectors
                writeln!(fo, "{} {}", vocab_size, layer1_size)?;
                for (a, vw) in self.vocab.iter().enumerate() {
                    write!(fo, "{} ", vw.word).context("error writing output file")?;
                    let word_vec = &self.syn0[a * layer1_size..][..layer1_size];
                    if self.options.binary {
                        let word_vec = word_vec.iter().map(Real::get).collect::<Vec<real>>();
                        fo.write_all(bytemuck::cast_slice::<real, u8>(&word_vec))
                            .context("error writing output file")?;
                    } else {
                        for f in word_vec {
                            write!(fo, "{} ", f.get()).context("error writing output file")?;
                        }
                        writeln!(fo).context("error writing output file")?;
                    }
                }
            }
            Some(classes) => {
                // Run K-means on the word vectors
                let clcn = classes;
                let mut centcn = vec![0i32; classes];
                let mut cl: Vec<usize> = (0..vocab_size).map(|a| a % clcn).collect();
                let mut cent: Vec<real> = vec![0.0; classes * layer1_size];

                let iter = 10;
                for _ in 0..iter {
                    cent.fill(0.0);
                    centcn[0..clcn].fill(1);

                    // Set cent[c] = sum of vectors in class c, centcn[c] = number of vectors in class c + 1
                    for c in 0..vocab_size {
                        for d in 0..layer1_size {
                            cent[layer1_size * cl[c] + d] += self.syn0[c * layer1_size + d].get();
                        }
                        centcn[cl[c]] += 1;
                    }

                    // Set cent[c] = center of class c, normalized to length 1
                    for b in 0..clcn {
                        let mut closev = 0.0;
                        for c in 0..layer1_size {
                            cent[layer1_size * b + c] /= centcn[b] as real;
                            closev += cent[layer1_size * b + c].powi(2);
                        }
                        closev = closev.sqrt();
                        for c in 0..layer1_size {
                            cent[layer1_size * b + c] /= closev;
                        }
                    }

                    // Move vectors to nearest class (by dot-product similarity with center of class).
                    for c in 0..vocab_size {
                        // (This could use max_by_key, but it would require ordered_float and would
                        // change NaN handling.)
                        let mut closev: real = -10.0;
                        let mut closeid = 0;
                        for d in 0..clcn {
                            let x = (0..layer1_size)
                                .map(|b| {
                                    cent[layer1_size * d + b] * self.syn0[c * layer1_size + b].get()
                                })
                                .sum::<real>();
                            if x > closev {
                                closev = x;
                                closeid = d;
                            }
                        }
                        cl[c] = closeid;
                    }
                }

                // Save the K-means classes
                for a in 0..vocab_size {
                    writeln!(fo, "{} {}", self.vocab[a].word, cl[a])
                        .context("error writing k-means classes to output file")?;
                }
            }
        }
        Ok(())
    }
}

fn main() {
    let options = Options::parse();

    let mut word3vec = Word3Vec::new(options);
    if let Err(err) = word3vec.train_model() {
        eprintln!("{err:#}");
        process::exit(1);
    }
}
