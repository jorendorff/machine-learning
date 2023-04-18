import re
from pathlib import Path
import random
import textwrap

from tensorflow import keras
from tensorflow.keras import layers, models, utils
import numpy as np

from corpus import corpus


def make_model(vocab_size, vec_size):
    return models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=vec_size),  # encoder
        layers.LSTM(vec_size, return_sequences=True),
        layers.Dense(vocab_size, activation='softmax'), # decoder
    ])


def prepare_model(corpus, vec_size):
    vocab_size = len(corpus.vocab)
    model = make_model(vocab_size, vec_size)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.build(input_shape=(1, len(corpus.vocab)))
    model.summary()

    if Path(SAVE_FILE).is_file():
        model.load_weights(SAVE_FILE)
    return model


class CorpusSequence(utils.Sequence):
    def __init__(self, corpus):
        self.vocab_size = len(corpus.vocab)
        self.eos = corpus.vocab_map["<EOS>"]

        samples = list(corpus.sentences)
        random.shuffle(samples)
        self.sentences = samples

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        # Returns two matrices, but each with just 1 row. Batch size is 1.
        sentence = self.sentences[i]
        v = self.vocab_size
        x = np.array([sentence], dtype=int)
        y = np.array([sentence[1:] + [self.eos]], dtype=int)
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


def babble(corpus, model, prompt):
    drivel = corpus.tokenize(prompt)
    start = len(drivel)

    eos = corpus.vocab_map['<EOS>']  # end-of-sentence tag

    nv = len(corpus.vocab)
    while len(drivel) < 50 and drivel[-1:] != [eos]:
        x = np.array([drivel], dtype=int)
        p = model(x, training=False)[0, -1]
        # Randomly sample a bunch of candidates. Choose the most likely of them.
        candidates = random_choice(nv, 3, p)
        next_word = max(candidates, key=lambda c: p[c])
        drivel.append(next_word)

    return prompt.upper() + " " + " ".join(corpus.vocab[i] for i in drivel[start:])


PROMPTS = [
    "The easiest way to",
    "In this chapter,",
    "Note that there are",
    "This works because the",
    "It's useful when you",
    "For example, the",
    "If an error occurs",
]


SAVE_FILE = "./word2vec-rnn-saved-weights.keras"


class Callbacks(keras.callbacks.Callback):
    def __init__(self, corpus):
        self.batch_count = 0
        self.corpus = corpus
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
                text = babble(self.corpus, self.model, prompt)
                print(textwrap.fill(text, initial_indent="-   ", subsequent_indent="    "))
                print()
            print()


def main():
    vec_size = 100
    model = prepare_model(corpus, vec_size)

    examples = CorpusSequence(corpus)
    my_callbacks = Callbacks(corpus)
    model.fit(examples, epochs=15, callbacks=my_callbacks)


if __name__ == '__main__':
    main()
