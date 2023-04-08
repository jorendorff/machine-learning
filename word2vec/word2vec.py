from tensorflow import keras
from tensorflow.keras import layers, models, utils
import numpy as np
import re
from pathlib import Path
import random
import textwrap


def make_model(vocab_size, vec_size):
    return models.Sequential([
        layers.Dense(
            vec_size,
            activation='relu',
        ),
        layers.Dense(
            vocab_size,
            activation='softmax',
        ),
    ])


def prepare_model(corpus, vec_size, context_len):
    vocab_size = len(corpus.vocab)
    model = make_model(vocab_size, vec_size)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.build(input_shape=(1, len(corpus.vocab) * context_len))

    if Path(SAVE_FILE).is_file():
        model.load_weights(SAVE_FILE)

    return model


def tokenize(text):
    return re.findall(r'[ \t]*(\n|[A-Za-z0-9\'_]+|.)', text.lower())


class Corpus:
    def __init__(self, dir):
        corpus = []
        for f in Path(dir).glob("*.md"):
            corpus += tokenize(f.read_text())
            if corpus[-1:] != ['\n']:
                corpus.append('\n')

        self.vocab = sorted(set(corpus))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        self.tokens = [self.vocab_map[word] for word in corpus]

    def tokenize(self, text):
        return [self.vocab_map[word] for word in tokenize(text)]

    def prompt_to_x(self, prompt):
        v = len(self.vocab)
        c = len(prompt)
        x = np.zeros((1, c * v))
        for j, token in enumerate(prompt):
            k = j * v + token
            x[0, k] = 1.0
        return x


class CorpusSequence(utils.Sequence):
    def __init__(self, corpus, context_len, batch_size):
        self.corpus = corpus
        self.context_len = context_len
        self.batch_size = batch_size

        # Example: if len(t) == 6 and context_len == 4, then
        # we can get 2 examples, using t[0:4] to predict t[4],
        # and using t[1:5] to predict t[5].
        num_examples = max(0, len(corpus.tokens) - context_len)
        starts = list(range(num_examples))
        random.shuffle(starts)
        n = context_len

        # Each batch is stored in the form of a python list of starting offsets.
        self.batches = [starts[i:i+n] for i in range(0, num_examples, n)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        starts = self.batches[i]
        corpus = self.corpus
        v = len(corpus.vocab)
        c = self.context_len

        x = np.zeros((len(starts), c * v), dtype=float)
        y = np.zeros((len(starts)), dtype=int)
        for row, start in enumerate(starts):
            predict = start + c
            for j, token in enumerate(corpus.tokens[start:predict]):
                k = j * v + token
                x[row, k] = 1.0
            y[row] = corpus.tokens[predict]

        return x, y


def random_choice(nt, num_samples, p):
    samples = []
    for i in range(num_samples):
        acc = np.cumsum(p)
        z = random.random()
        next_word = np.sum(acc < z)
        if next_word == nt:
            next_word -= 1
        samples.append(next_word)
    return samples


def babble(corpus, context_len, model, prompt):
    drivel = corpus.tokenize(prompt)
    nl = corpus.vocab_map['\n']
    dot = corpus.vocab_map['.']

    nt = len(corpus.tokens)
    c = context_len
    assert len(drivel) >= c
    while len(drivel) < 100 and drivel[-1:] != [dot] and drivel[-2:] != [nl, nl]:
        x = corpus.prompt_to_x(drivel[-c:])
        p = model(x, training=False)[0]
        # Randomly sample a bunch of candidates. Choose the most likely of them.
        candidates = random_choice(nt, 10, p)
        next_word = max(candidates, key=lambda c: p[c])
        drivel.append(next_word)

    return " ".join(corpus.vocab[i] for i in drivel)


PROMPTS = [
    "The easiest way to",
    "In this chapter,",
    "Note that there are",
    "This works because the",
    "It's useful when you",
    "For example, the",
    "If an error occurs",
]


SAVE_FILE = "./word2vec-saved-weights.keras"


class Callbacks(keras.callbacks.Callback):
    def __init__(self, corpus, context_len):
        self.batch_count = 0
        self.corpus = corpus
        self.context_len = context_len
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1

        if self.batch_count % 1000 == 0:
            self.model.save_weights(SAVE_FILE)

        if self.batch_count % 1000 == 1:
            print()
            print(f"After {self.batch_count} batches:")
            print()
            for prompt in PROMPTS:
                text = babble(self.corpus, self.context_len, self.model, prompt)
                print(textwrap.fill(text, initial_indent="-   ", subsequent_indent="    "))
                print()
            print()


def main():
    DIR = '/home/jorendorff/src/rust-book/atlas'
    corpus = Corpus(DIR)
    vec_size = 100
    context_len = 4
    model = prepare_model(corpus, vec_size, context_len)

    examples = CorpusSequence(corpus, context_len, batch_size=64)
    my_callbacks = Callbacks(corpus, context_len)
    model.fit(examples, epochs=1, callbacks=my_callbacks)


if __name__ == '__main__':
    main()
