import csv
import numpy as np
import random
import tensorflow as tf

from params import EMBEDDING_SIZE

learning_rate = 0.01
num_epoch = 100
batch_size = 128
num_class = 2  # spam (1 or one-hot [0, 1]) or ham (0 or one-hot [1, 0])
num_input = EMBEDDING_SIZE  # sentence embedding size
num_hidden_1 = 64
num_hidden_2 = 128
num_hidden_3 = 32
train_data_ratio = 0.8

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_class])


def network():
    layer_1 = tf.layers.dense(X, num_hidden_1, activation=tf.nn.relu, name='dense_1')
    layer_1 = tf.layers.dropout(layer_1, name='dropout_1')

    layer_2 = tf.layers.dense(layer_1, num_hidden_2, activation=tf.nn.relu, name='dense_2')
    layer_2 = tf.layers.dropout(layer_2, name='dropout_2')

    layer_3 = tf.layers.dense(layer_2, num_hidden_3, activation=tf.nn.relu, name='dense_3')
    layer_3 = tf.layers.dropout(layer_3, name='dropout_3')

    output = tf.layers.dense(layer_3, num_class, name='output_layer')

    return output


def loss(logits, labels):
    loss_ops = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_ops = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ops)
    return loss_ops, train_ops


out = network()
loss_op, train_op = loss(out, Y)
correct_predictions = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def read_data():
    data_array = []
    with open('./data/enron_email_vectors_50d.csv', newline='') as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in csvreader:
            vector = np.array(row[2].split(' '), dtype=np.double)
            data_array.append([int(row[1]), np.array(vector, dtype=np.double)])
    return data_array


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def split_data(data_array: list, train_to_all_ration: float):
    num_train = int(len(data_array) * train_to_all_ration)
    data_train = data_array[:num_train]
    data_test = data_array[num_train:]
    return data_train, data_test


def get_features_and_labels(data_array: list):
    labels = [item[0] for item in data_array]
    features = [item[1] for item in data_array]
    return np.array(features, dtype=np.double), get_one_hot(np.array(labels, dtype=np.int), num_class)


def batch_generator(data_array: list):
    data_len = len(data_array)
    steps_per_epoch = int(data_len / batch_size) + 1
    for step in range(steps_per_epoch):
        if step == steps_per_epoch - 1:
            yield get_features_and_labels(
                data_array[step * batch_size:] + data_array[:data_len - step * batch_size])
        else:
            yield get_features_and_labels(data_array[step * batch_size: (step + 1) * batch_size])


data = read_data()
print(len(data))
train_data, test_data = split_data(data, 0.8)
print(len(train_data), len(test_data))
X_test, Y_test = get_features_and_labels(test_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    global_step = 1
    for epoch in range(1, num_epoch + 1):
        print("Epoch: " + str(epoch))
        for batch_x, batch_y in batch_generator(train_data):
            loss, acc, _ = sess.run([loss_op, accuracy, train_op],
                                    feed_dict={X: batch_x, Y: batch_y})
            global_step += 1
        print("End of epoch " + str(epoch)
              + ", batch loss=" + "{:.4f}".format(loss)
              + ", batch accuracy=" + "{:.3f}".format(acc))
        random.shuffle(train_data)
    print("Training finished on step " + str(global_step))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))
