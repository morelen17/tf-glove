import collections
import numpy as np
import os
import re
import tensorflow as tf
import time

from helpers import read_emails, save_dict, load_dict
from params import EMBEDDING_SIZE, PATH_TO_ENRON_DATASET


class Glove:
    def __init__(self):
        self._embedding_size = EMBEDDING_SIZE
        self._x_max = 100.0
        self._alpha = 0.75
        self._window_size = 10
        self._vocabulary = collections.Counter()
        self._vocab_size = 25000
        self._cooccur_matrix = []

        self._batch_size = 128
        self._learning_rate = 0.001
        self._num_epoch = 10

        self._i_idx, self._j_idx, self._x_ij = None, None, None
        self._w_1, self._w_2 = None, None
        self._loss_op, self._train_op = None, None

    def _get_vocab_from_file(self):
        path = './data/vocab_25k_enron.pkl'
        if os.path.exists(path):
            self._vocabulary = load_dict(path)
            return True
        return False

    def _get_cooccur_matrix_from_file(self):
        path = './data/cooccurrence_25k_enron.npy'
        if os.path.exists(path):
            self._cooccur_matrix = np.load(path)
            return True
        return False

    def build_vocab(self, data_folder: str):
        if not self._get_vocab_from_file():
            pattern = re.compile('[^a-z ]+')
            for file in read_emails(data_folder):
                path_to_file = os.path.join(data_folder, file)
                with open(path_to_file, 'rb') as f:
                    text = f.read()
                text2 = pattern.sub('', text.decode('utf8', 'ignore').lower().replace('\n', ' '))
                for word in text2.split(' '):
                    if word != '':
                        self._vocabulary[word] += 1
            self._vocabulary = self._vocabulary.most_common(self._vocab_size)  # [ ('the', 290813), (...), ...]
            self._vocabulary = {self._vocabulary[idx][0]: idx for idx in range(self._vocab_size)}
        return self._vocabulary

    def build_cooccur_matrix(self, path_to_folder: str):
        if not self._get_cooccur_matrix_from_file():
            pattern = re.compile('[^a-z ]+')
            cooc_mat = np.zeros((self._vocab_size, self._vocab_size), dtype=np.float32)
            for file in read_emails(path_to_folder):
                path_to_file = os.path.join(path_to_folder, file)
                with open(path_to_file, 'rb') as f:
                    text = f.read()
                text2 = pattern.sub('', text.decode('utf8', 'ignore').lower().replace('\n', ' '))
                words = text2.split(' ')
                words_len = len(words)
                for i in range(words_len):
                    if words[i] in self._vocabulary:
                        idx = self._vocabulary[words[i]]
                        for j in range(1, self._window_size + 1):
                            if i - j > 0:
                                if words[i - j] in self._vocabulary:
                                    l_idx = self._vocabulary[words[i - j]]
                                    cooc_mat[idx, l_idx] += 1.0 / j
                            if i + j < words_len:
                                if words[i + j] in self._vocabulary:
                                    r_idx = self._vocabulary[words[i + j]]
                                    cooc_mat[idx, r_idx] += 1.0 / j
            self._cooccur_matrix = cooc_mat
        return self._cooccur_matrix

    def build(self):
        x_max = tf.constant(self._x_max, dtype=tf.float32)
        alpha = tf.constant(self._alpha, dtype=tf.float32)

        self._i_idx = tf.placeholder(tf.int32, shape=[self._batch_size])  # i indexes
        self._j_idx = tf.placeholder(tf.int32, shape=[self._batch_size])  # j indexes
        self._x_ij = tf.placeholder(tf.float32, shape=[self._batch_size])  # X_ij

        self._w_1 = tf.Variable(tf.random_uniform([self._vocab_size, self._embedding_size], 1.0, -1.0))
        self._w_2 = tf.Variable(tf.random_uniform([self._vocab_size, self._embedding_size], 1.0, -1.0))
        b_1 = tf.Variable(tf.random_uniform([self._vocab_size], 1.0, -1.0))
        b_2 = tf.Variable(tf.random_uniform([self._vocab_size], 1.0, -1.0))

        ww_1 = tf.nn.embedding_lookup([self._w_1], self._i_idx)
        ww_2 = tf.nn.embedding_lookup([self._w_2], self._j_idx)
        bb_1 = tf.nn.embedding_lookup([b_1], self._i_idx)
        bb_2 = tf.nn.embedding_lookup([b_2], self._j_idx)

        log_x_ij = tf.log(self._x_ij)
        weighted_x = tf.minimum(1.0, tf.pow(tf.div(self._x_ij, x_max), alpha))

        distance = tf.square(tf.add_n([
            tf.reduce_sum(tf.multiply(ww_1, ww_2), 1),
            bb_1,
            bb_2,
            tf.negative(log_x_ij)
        ]))
        self._loss_op = tf.reduce_sum(tf.multiply(weighted_x, distance))
        self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss_op)
        return

    def _batch_generator(self):
        cooccurrences = np.transpose(np.nonzero(self._cooccur_matrix))  # indexes of non-zero elements
        steps_per_epoch = int(len(cooccurrences) / self._batch_size) + 1
        for step in range(steps_per_epoch):
            idxs = np.random.choice(len(cooccurrences), self._batch_size)
            idx_i, idx_j, x_ij = [], [], []
            for coord in cooccurrences[idxs]:
                idx_i.append(coord[0])
                idx_j.append(coord[1])
                x_ij.append(self._cooccur_matrix[coord[0], coord[1]])
            yield idx_i, idx_j, x_ij

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 1
            for epoch in range(1, self._num_epoch + 1):
                epoch_loss = 0
                print("Epoch:", str(epoch))
                for idx_i, idx_j, x_ij in self._batch_generator():
                    loss, _ = sess.run([self._loss_op, self._train_op], feed_dict={
                        self._i_idx: idx_i,
                        self._j_idx: idx_j,
                        self._x_ij: x_ij
                    })
                    epoch_loss += loss
                    if global_step % 1000 == 0:
                        print("Step", str(global_step), '| Batch loss =', "{:.4f}".format(epoch_loss / 1000))
                        epoch_loss = 0
                    global_step += 1
                print("End of epoch %d (step %d)\n-------" % (epoch, global_step))
            return self._w_1.eval(sess), self._w_2.eval(sess)


if __name__ == '__main__':
    glove = Glove()
    t = time.time()
    vocab = glove.build_vocab(data_folder=PATH_TO_ENRON_DATASET)
    print('Vocabulary build completed! Time, s:', (time.time() - t))
    # save_dict(vocab, './data/vocab_25k_enron.pkl')
    # print('Vocabulary saved!')

    t = time.time()
    coocur_matrix = glove.build_cooccur_matrix(path_to_folder=PATH_TO_ENRON_DATASET)
    print('Co-occurrence matrix build completed! Time, s:', (time.time() - t))
    # np.save('./data/cooccurrence_25k_enron.npy', coocur_matrix)
    # print('Co-occurrence matrix saved!')

    glove.build()
    print('Model build completed!')
    w1, w2 = glove.train()
    print('Train completed!')
    avg_w = (w1 + w2) / 2
    t = time.time()
    with open('./data/enron_glove_word_vectors_50d.txt', 'w') as f:
        for word, idx in vocab.items():
            str_line = word + ' ' + ' '.join(str(x) for x in avg_w[idx].tolist()) + '\n'
            f.write(str_line)
    print('Word vectors saved! Time, s:', (time.time() - t))
