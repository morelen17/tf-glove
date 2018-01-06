import codecs
import csv
import glob
import json
import numpy as np
import os
import re
import time

from params import EMBEDDING_SIZE, PATH_TO_ENRON_DATASET
from sif import SIF
from six.moves import cPickle


def process_vector_line(line: str, pattern):
    values = line.split(' ')
    word = values[0]
    if pattern.match(word) is not None:
        vector = np.array(values[1:], dtype=np.double)
        return [word, vector]
    pass


def process_vectors_text(path_to_file: str):
    dict_to_save = {}
    regexp = re.compile(r'^[a-zA-Z]*$')
    for line in open(path_to_file):
        parsed = process_vector_line(line, regexp)
        if parsed is not None:
            dict_to_save[parsed[0]] = parsed[1]
    return dict_to_save


def save_dict(data: dict, path: str):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
    pass


def load_dict(path: str) -> dict:
    with open(path, 'rb') as f:
        data = cPickle.load(f)
    return data


def read_email_to_array(path_to_file: str, words_dict: dict) -> list:
    words_array = []
    pattern = re.compile('[^a-zA-Z]')
    for line in open(path_to_file, 'rb'):
        for word in line.decode('utf8', 'ignore').split(' '):
            word = pattern.sub('', word.lower())
            if word in words_dict:
                words_array.append(word)
    return words_array


def read_emails(folder: str):
    for file in glob.iglob(folder + '**/*.txt', recursive=True):
        if file.endswith('am.txt'):
            yield file


def process_emails(folder: str, words_dict: dict):
    email_arr = []
    sif = SIF(words_dict)
    for file in read_emails(folder):
        path_to_file = os.path.join(folder, file)
        is_spam = int(file.endswith('spam.txt'))
        email_vector = sif.get_sentence_vector(read_email_to_array(path_to_file, words_dict))
        email_arr.append([os.path.basename(file), is_spam, email_vector])
    return email_arr


def save_emails_to_csv(csv_to_save: str, email_list: list):
    with open(csv_to_save, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=';')
        for email in email_list:
            csv_writer.writerow([email[0], email[1], ' '.join(str(e) for e in email[2].tolist())])
    pass


def to_json(out_path: str, vocab: dict):
    json_map = {
        'vocabulary': vocab,
        'vocabularySize': len(vocab),
        'vectorSize': EMBEDDING_SIZE
    }
    json.dump(json_map,
              codecs.open(out_path, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)
    return


if __name__ == '__main__':
    t = time.time()
    # vocabulary = load_dict('./data/vocab_25k_enron.pkl')
    vocabulary = process_vectors_text('./data/enron_glove_word_vectors_50d.txt')
    print('Word vectors processed! Time, s:', (time.time() - t))

    # t = time.time()
    # emails = process_emails(PATH_TO_ENRON_DATASET, vocabulary)
    # print('Emails processed! Time, s:', (time.time() - t))

    # t = time.time()
    # save_emails_to_csv('./data/enron_email_vectors_50d.csv', emails)
    # print('Email vectors saved! Time, s:', (time.time() - t))

    t = time.time()
    to_json('./data/enron_glove_word_vectors_50d.json', {w: v.tolist() for w, v in vocabulary.items()})
    print('Vectors was saved to json! Time, s:', (time.time() - t))
