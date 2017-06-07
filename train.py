# -*- coding: UTF-8 -*-

import tensorflow as tf
from graph import CpRModel
from data import Data
import argparse
from prepare import Vocabulary
import cPickle as pickle
# import predict

def train_per_epoch(index, model, data, batch_size, sample_size, with_rnn, alpha):
    saver = tf.train.Saver(tf.global_variables())
    savePath = 'exp/batch_%d.len_%d.rnn_%s.alpha_%f.ckpt' % \
                (batch_size, sample_size, str(with_rnn), alpha)

    with tf.Session() as sess:
        try:
            saver.restore(sess, savePath)
        except:
            print 'wrong!'
            sess.run(tf.global_variables_initializer())

        epoch_size = (data.iter / batch_size) / 100
        for i in xrange(epoch_size):
            ar, lr = get_lr(epoch_size, i)
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input: x_batch, model.output: y_batch, model.learning_rate: lr, model.anneal_rate: ar}
            sess.run([model.cost, model.optimizer, model.last_state], feed_dict)

            if i % (epoch_size / 100) == 0:
                cost = sess.run(model.cost, feed_dict)
                print 'epoch %03i, %d steps, %.2f of all done, cost = %.3f' % \
                        (index, i, ar, cost)

                vocab = pickle.load(open('data/ch_char_vocab.pkl'))
                ss = ''
                for i in xrange(sample_size):
                    ss += vocab.by_index(x_batch[0, i])
                print 'x_batch = ', ss
                ss = ''
                for i in xrange(sample_size):
                    ss += vocab.by_index(y_batch[0, i])
                print 'y_batch = ', ss
                ss = ''
                result = sess.run(model.result, feed_dict)
                for i in xrange(sample_size):
                    ss += vocab.by_index(result[0, i])
                print 'result = ', ss

                # predict.main(n, batch_size, sample_size, with_rnn, alpha)

        saver.save(sess, savePath)

def get_lr(whole, current, learning_rate=0.01, decay_rate=0.95, decay_after=0.7, decay_step=0.1):
    rc = float(current) / whole
    if rc < decay_after:
        return rc, learning_rate
    else:
        lr = learning_rate * decay_rate ** ((rc - decay_after) // decay_step)
        return rc, lr

def main(epochs, batch_size, sample_size, with_rnn, alpha):
    print '************* training start! *************'
    data = Data(sample_size=sample_size, batch_size=batch_size)
    model = CpRModel(batch_size, sample_size, with_rnn=with_rnn, alpha=alpha)
    for i in xrange(epochs):
        train_per_epoch(i, model, data, batch_size, sample_size, with_rnn, alpha)
    print '************* training end! **************'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-sample_size', default=100, type=int)
    parser.add_argument('-with_rnn', default=True, type=bool)
    parser.add_argument('-alpha', default=0.2, type=float)
    args = parser.parse_args()
    main(**vars(args))