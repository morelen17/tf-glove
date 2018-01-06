import numpy as np

from params import EMBEDDING_SIZE
from sklearn.decomposition import PCA
from wordfreq import word_frequency


class SIF:
    def __init__(self, vocab: dict):
        self._vocabulary = vocab
        self._a = 0.001
        self._embedding_size = EMBEDDING_SIZE
        self._language = 'en'
        self._singular_vector = None

    def get_sentence_vector(self, word_array: list):
        vector = np.zeros(self._embedding_size)
        sentence_len = 0
        for item in word_array:
            if item in self._vocabulary:
                weight = self._a / (self._a + word_frequency(item, self._language))
                vector = np.add(vector, np.multiply(weight, self._vocabulary[item]))
                sentence_len += 1
        vector = np.divide(vector, sentence_len)
        return vector

    def compute_singular_vector(self, sentence_matrix):
        pca = PCA(n_components=self._embedding_size)
        pca.fit(np.array(sentence_matrix).transpose())
        u_vector = pca.components_[0]
        self._singular_vector = np.matmul(u_vector, np.transpose(u_vector))
        return self._singular_vector
