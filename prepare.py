# -*- coding: UTF-8 -*-

import argparse
import cPickle as pickle
import numpy as np

dataset = "ch_entertainment"

class Vocabulary:

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.idx = 0

    def add(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        self.word_to_index[word] = self.idx
        self.index_to_word[self.idx] = word
        self.idx += 1
        return self.idx - 1

    def by_word(self, word, oov_word=None):
        if word in self.word_to_index:
            return self.word_to_index[word]
        if oov_word is not None:
            assert oov_word in self.word_to_index
            return self.word_to_index[oov_word]
        return -1

    def by_index(self, index):
        try:
            return self.index_to_word[index]
        except:
            print "vocab.by_index: index out of range ", index
            return ' '

    def __len__(self):
        assert len(self.index_to_word) == len(self.word_to_index)
        return len(self.index_to_word)

def make_vocab():
    char_vocab = Vocabulary()
    with open('data/%s/yule.txt' % (dataset)) as f:
        chars = f.read().decode('utf-8')
    for i in xrange(len(chars)):
        ch = chars[i]
        char_vocab.add(ch)
        if i % 10000000 == 0:
            print "\r%d of %d" % (i, len(chars))

    with open("data/ch_char_vocab.pkl", "w") as wf:
        pickle.dump(char_vocab, wf)

    return chars

def convert(phase, raw_chars):
    vocab = pickle.load(open("data/ch_char_vocab.pkl", 'rb'))
    chars = np.zeros((len(raw_chars)))
    for i in xrange(len(raw_chars)):
        chars[i] = vocab.by_word(raw_chars[i])
    np.save("data/ch.%s.npy" % phase, chars.astype('int32'))
    print phase, "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='ch_entertainment')
    args = parser.parse_args()
    dataset = args.dataset
    print dataset

    t = make_vocab()

    convert("train", t)
    # convert("valid", v)

    assert np.max(np.load('data/ch.train.npy')) == len(pickle.load(open("data/ch_char_vocab.pkl", 'rb')).index_to_word)-1
