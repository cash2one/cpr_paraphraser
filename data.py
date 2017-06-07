# -*- coding: UTF-8 -*-

import numpy as np
import cPickle as pickle
from prepare import Vocabulary

class Data:

    def __init__(self, sample_size, batch_size):
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.dataset = np.load('data/ch.%s.npy' % 'train')
        print np.shape(self.dataset)[0]
        self.vocab = pickle.load(open('data/ch_char_vocab.pkl'))
        
        judge_ends = ['。', '！', '？']
        self.sente_end = []
        for c in judge_ends:
            i = self.vocab.by_word(c.decode('utf-8'))
            self.sente_end.append(i)

        self.iter = self.count_sample()
        self.newline = self.vocab.by_word("\n")
        self.pad_word = self.vocab.by_word(' ')

    def count_sample(self):
        count = 0
        for char in self.sente_end:
            c = len(np.where(self.dataset==char)[-1])
            print 'number of %s : %d' % (self.vocab.by_index(char), c)
            count += c
        print 'sentence number: ', count

        return count

    def next_batch(self):
        indices = np.random.randint(0, self.dataset.shape[0] - self.sample_size, self.batch_size)
        x = np.zeros((self.batch_size, self.sample_size))
        y = np.zeros((self.batch_size, self.sample_size))
        for i in xrange(self.batch_size):
            sample_x, sample_y = self.get_sample(indices[i])
            x[i, :] = sample_x
            y[i, :] = sample_y

        # ss = ''
        # for i in xrange(self.sample_size):
        #     ss += self.vocab.by_index(x[0, i])
        # print ss
        # print y
        return x.astype('int32'), y.astype('int32')

    def get_sample(self, idx):
        while True:
            while True:      
                temp_char = self.dataset[idx]
                if temp_char in self.sente_end:
                    idx += 1
                    break
                else:
                    idx -= 1
            s = self.dataset[idx: idx+self.sample_size]
            cut_idx = []
            # for i in xrange(len(s)):
            #     if s[i] in self.sente_end + [self.newline]:
            #         cut_idx.append(i)
            # if len(cut_idx) != 0:
            #     s[cut_idx[0]+1:] = self.pad_word

            if len(s) == self.sample_size and (len(cut_idx)==0 or cut_idx[0] > 2):
                o = self.dataset[idx+1: idx+self.sample_size+1]
                # o = np.zeros(np.shape(s), dtype='int32')
                # o[:self.sample_size-1] = s[1:]
                # o[-1] = self.pad_word
                return s, o
            else:
                idx = np.random.randint(0, self.dataset.shape[0] - 2 * self.sample_size)
