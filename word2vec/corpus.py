""" Corpus: sentences from _Programming Rust_. """

import re
import random
from pathlib import Path
from collections import defaultdict, Counter

import marko.parser


def tokenize(text):
    return re.findall(r'[ \t]*(\n|c\+\+|(?:[A-Za-z0-9\'_]|\.[a-z])+|.)', text.lower())


MARKO_NONLEAF_INLINE_CLASSES = (
    marko.inline.Emphasis,
    marko.inline.StrongEmphasis,
    marko.inline.Link,
)


def _paragraph_words(para):
    def render_inline(elem):
        if isinstance(elem, marko.inline.RawText):
            return tokenize(elem.children.strip())
        elif isinstance(elem, MARKO_NONLEAF_INLINE_CLASSES):
            return _paragraph_words(elem)
        elif isinstance(elem, marko.inline.CodeSpan):
            return ["`" + elem.children]
        elif isinstance(elem, marko.inline.AutoLink):
            return ['<LINK>']
        elif isinstance(elem, marko.inline.LineBreak):
            return []
        elif isinstance(elem, marko.inline.InlineHTML) and elem.children.startswith("<index "):
            return []
        else:
            return ['<UNK>']

    words = []
    for c in para.children:
        words += render_inline(c)
    return words


class Corpus:
    def __init__(self, dir):
        sentences = []

        for f in Path(dir).glob("*.md"):
            doc = marko.Markdown().parse(f.read_text())
            for elem in doc.children:
                if isinstance(elem, marko.block.Paragraph):
                    words = _paragraph_words(elem)
                    while words:
                        if words[0] in ('(', ')'):
                            del words[0]
                            continue
                        n = len(words)
                        for punct in ('.', '?', '!'):
                            if punct in words:
                                n = min(n, words.index(punct) + 1)
                        sentences.append(words[:n])
                        del words[:n]

        word_counts = Counter(word for s in sentences for word in s)
        sentences = [
            [(w if word_counts[w] > 1 else '<UNK>') for w in s]
            for s in sentences
        ]
        self.vocab = sorted(set(['<EOS>'] + [w for s in sentences for w in s]))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        self.sentences = [
            [self.vocab_map[w] for w in s]
            for s in sentences
        ]

    def tokenize(self, text):
        return [self.vocab_map[word] for word in tokenize(text)]


DIR = '/home/jorendorff/src/rust-book/atlas'
corpus = Corpus(DIR)
