# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import cPickle as pickle
import argparse
from graph import CpRModel
from prepare import Vocabulary

def to_inputs(si, vocab, sample_size):
    assert len(si) <= sample_size
    s = [vocab.by_word(w) for w in si.decode('utf-8')]
    for i in xrange(sample_size - len(s)):
        s.append(vocab.by_word(' '))
    return np.asarray(s)

def predict(model, input, n, batch_size, sample_size, with_rnn, alpha):
    vocab = pickle.load(open('data/ch_char_vocab.pkl'))
    input = to_inputs(input, vocab, sample_size)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        savePath = 'exp/batch_%d.len_%d.rnn_%s.alpha_%f.ckpt' % \
                (batch_size, sample_size, str(with_rnn), alpha)

        try:
            saver.restore(sess, savePath)
        except:
            raise IOError('no such model')

        feed_dict = {model.input: np.tile(input, n).reshape([n, sample_size])}
        output = sess.run(model.result, feed_dict=feed_dict)

    print 'output = ', output
    results = []
    for i in xrange(n):
        sen = ''
        for j in xrange(sample_size):
            sen += vocab.by_index(output[i][j])
        results.append(sen)

    return results

def main(n, batch_size, sample_size, with_rnn, alpha):
    print '************* predicting start! *************'
    
    input = '贾樟柯希望把平遥国际电影展打造成一个“小身段大格局”的影展'
    print 'origin = ', input

    model = CpRModel(n, sample_size, with_rnn=with_rnn, alpha=alpha)
    results = predict(model, input, n, batch_size, sample_size, with_rnn, alpha)
    
    print 'paraphrased = '
    for sentence in results:
        print sentence

    print '************* predicting end! **************'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=15, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-sample_size', default=100, type=int)
    parser.add_argument('-with_rnn', default=True, type=bool)
    parser.add_argument('-alpha', default=0.2, type=float)
    args = parser.parse_args()
    main(**vars(args))