import math
import textwrap

import numpy as np
import keras_nlp

import word2vec_rnn
import corpus
from corpus import corpus as C

vec_size = 100
model = word2vec_rnn.prepare_model(C, vec_size)

w, b = model.layers[-1].get_weights()
vocab_size = len(C.vocab)
assert w.shape == (vec_size, vocab_size)
assert b.shape == (vocab_size,)
vocab = C.vocab

# which words have greatest magnitude? which the least?

print("50 words with greatest 'bias':")
by_b = sorted(range(vocab_size), key=lambda i: b[i], reverse=True)
for i, word in enumerate(by_b[:50]):
    print(f"{i + 1}.  {C.vocab[word]}   {b[word]:.3f}")
print()


def magnitude(i):
    return np.linalg.norm(w[:,i])


print("Top 50 words by vector magnitude:")
by_w2 = sorted(range(vocab_size), key=magnitude, reverse=True)
for i, word in enumerate(by_w2[:50]):
    print(f"{i + 1}.  {C.vocab[word]}   {magnitude(word):.3f}")
print()


# Normalize all vectors to magnitude 1
w /= np.linalg.norm(w, axis=0)


def distance(u, v):
    assert len(u.shape) == 1
    assert u.shape == v.shape
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return 180/math.pi * math.acos(min(np.dot(u, v), 1.0))


def word_to_vec(word):
    if word not in C.vocab_map:
        raise ValueError(f"unknown word: {word!r}")
    wi = C.vocab_map[word]
    return w[:,wi]


def nearest(v):
    pairs = sorted(
        [(i, distance(v, w[:,i])) for i in range(vocab_size)],
        key=lambda pair: pair[1]
    )
    return pairs[:20]


class Parser:
    def __init__(self, code):
        self.code = code
        self.i = 0

    def at_end(self):
        return self.i == len(self.code)

    def match(self, token):
        if not self.at_end() and self.code[self.i] == token:
            self.i += 1
            return True
        else:
            return False

    def require_match(self, token):
        if not self.match(token):
            raise ValueError(f"expected {token!r}")

    def atom(self):
        if self.match('('):
            v = self.sum()
            self.require_match(')')
            return v
        elif self.at_end():
            raise ValueError("end of input reached")
        elif self.match('$'):
            self.require_match('plural')
            return PLURAL_AVG
        else:
            word = self.code[self.i]
            self.i += 1
            return word_to_vec(word)

    def sum(self):
        acc = self.atom()
        while True:
            if self.match('+'):
                acc = acc + self.atom()
            elif self.match('-'):
                acc = acc - self.atom()
            else:
                break
        return acc

    def parse(self):
        z = self.sum()
        if not self.at_end():
            raise ValueError(f"unexpected {self.code[self.i]!r}")
        return z


def vec_eval(words):
    return Parser(words).parse()


def print_nearest(v):
    for idx, (wi, d) in enumerate(nearest(v)):
        print(f"{idx + 1}.  {C.vocab[wi]}   {d:.1f}°")


PLURALS = [
    ('type', 'types'),
    ('function', 'functions'),
    ('method', 'methods'),
    ('value', 'values'),
    ('reference', 'references'),
    ('error', 'errors'),
    ('thread', 'threads'),
    ('pointer', 'pointers'),
    ('trait', 'traits'),
    ('file', 'files'),
]


PLURAL_AVG = sum(word_to_vec(p) - word_to_vec(s) for s, p in PLURALS) / len(PLURALS)


def main():
    while True:
        try:
            command = input("> ")
        except EOFError as _:
            print()
            break

        words = corpus.tokenize(command)

        if len(words) == 0:
            pass
        elif words[0] == ':eval':
            try:
                v = vec_eval(words[1:])
            except Exception as exc:
                print(exc)
                continue
            print_nearest(v)
        elif words[0] == ':w':
            [word] = words[1:]
            if word not in C.vocab_map:
                print("word not found")
                continue
            v = word_to_vec(word)
            print_nearest(v)
        elif words[0] == ':babble':
            del words[0]
            x = np.array([[C.vocab_map[word] for word in words]], dtype=int)
            p = model(x, training=False)[0, -1].numpy().tolist()
            pairs = sorted([(i, p[i]) for i in range(vocab_size)], key=lambda pair: pair[1], reverse=True)
            for idx, (wi, p) in enumerate(pairs[:5]):
                print(f"{idx + 1}.  {C.vocab[wi]}   P={p:.5f}")

            text = word2vec_rnn.babble(C, model, command)
            print(textwrap.fill(text, initial_indent="-   ", subsequent_indent="    "))
            print()
        elif words[0].startswith(':'):
            print("i didn't understand that")
        else:
            for word in words:
                if word not in C.vocab_map:
                    print(f"warning: unrecognized word: {word!r}")

            unk = C.vocab_map['<UNK>']
            prompt = [[C.vocab_map.get(word, unk) for word in words]]

            def token_probability_fn(x):
                return model(x, training=False)[:,-1]

            eos = C.vocab_map['<EOS>']  # end-of-sentence tag
            tokens = keras_nlp.utils.beam_search(
                token_probability_fn,
                prompt,
                max_length=50,
                num_beams=5,
                end_token_id=eos
            )
            tokens = tokens.numpy().tolist()
            if eos in tokens:
                del tokens[tokens.index(eos):]

            text = " ".join(C.vocab[i] for i in tokens)
            print(textwrap.fill(text))
            print()


if __name__ == '__main__':
    main()
