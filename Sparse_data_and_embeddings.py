#!usr/bin/python3
# -*- coding: UTF-8 -*-


import collections
import math

from IPython import display
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)

train_path = 'train.tfrecord'
test_path = 'test.tfrecord'


def parse_features_and_labels(record):

    features = {
        'terms': tf.VarLenFeature(dtype=tf.string),
        'labels': tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels  # 输出字典形式featurs，以便下面直接使用map函数调用


def my_input_fn(input_filenames, batch_size=1, shuffle=True, num_epochs=None):

    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(parse_features_and_labels)
    ds = ds.padded_batch(batch_size, ds.output_shapes).repeat(num_epochs)  # ds.output_shapes=({'terms': TensorShape([Dimension(None)])}, TensorShape([Dimension(1)])), Dimesion(None)意思是可变长度

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family", "man", "woman", "boy", "girl")

terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key='terms', vocabulary_list=informative_terms)  # 创建基于给定词汇表的分类标识列


def train_Linear_Classifer_medel(learning_rate, steps, batch_size, feature_columns, train_path, test_path):

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifer = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=my_optimizer)

    training_input_fn = lambda :my_input_fn(train_path , batch_size=batch_size)
    test_input_fn = lambda :my_input_fn(test_path, shuffle=False, num_epochs=1)

    linear_classifer.train(input_fn=training_input_fn, steps=steps)

    train_evaluation_metrics = linear_classifer.evaluate(input_fn=training_input_fn, steps=steps)
    test_evaluation_metrics = linear_classifer.evaluate(input_fn=test_input_fn, steps=steps)

    print('Training set metrics:')
    for m in train_evaluation_metrics:
        print(m, train_evaluation_metrics[m])

    print('----------------------')

    print('Test set metrics:')
    for m in test_evaluation_metrics:
        print(m, test_evaluation_metrics[m])

    return linear_classifer


# linear_classifier = train_Linear_Classifer_medel(
#     learning_rate=0.1,
#     steps=1000,
#     batch_size=25,
#     feature_columns=[terms_feature_column],
#     train_path=train_path,
#     test_path=test_path
# )


def train_DNN_Classifier_model(learning_rate, steps, batch_size, hidden_units, feature_columns, train_path, test_path):

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_classifer = tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns, optimizer=my_optimizer)

    training_input_fn = lambda :my_input_fn(train_path, batch_size=batch_size)
    test_input_fn = lambda :my_input_fn(test_path, shuffle=False, num_epochs=1)

    dnn_classifer.train(input_fn=training_input_fn, steps=steps)

    training_evaluation = dnn_classifer.evaluate(input_fn=training_input_fn, steps=steps)
    test_evaluation = dnn_classifer.evaluate(input_fn=test_input_fn, steps=steps)

    print('Training metrics:')
    for m in training_evaluation:
        print(m, training_evaluation[m])

    print('-----------------')

    print('Test metrics:')
    for m in test_evaluation:
        print(m, test_evaluation[m])

    return dnn_classifer


# dnn_classifier = train_DNN_Classifier_model(
#     learning_rate=0.1,
#     steps=1000,
#     batch_size=25,
#     hidden_units=[10, 10],
#     feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],
#     train_path=train_path,
#     test_path=test_path
# )

dnn_classifier_by_embedding = train_DNN_Classifier_model(
    learning_rate=0.1,
    steps=1000,
    batch_size=25,
    hidden_units=[10, 10],
    feature_columns=[tf.feature_column.embedding_column(terms_feature_column, dimension=2)],
    train_path=train_path,
    test_path=test_path
)

print(dnn_classifier_by_embedding.get_variable_names())
print(dnn_classifier_by_embedding.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape)

embedding_matrix = dnn_classifier_by_embedding.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

for term_index in range(len(informative_terms)):
    term_vector = np.zeros(len(informative_terms))
    term_vector[term_index] = 1
    embedding_xy = np.matmul(term_vector, embedding_matrix)  # 以上三行是为了依次取出embedding_matrix的各行值，等价于 embedding_matrix[term_index]
    plt.text(embedding_xy[0], embedding_xy[1], informative_terms[term_index])

plt.rcParams['figure.figsize'] = (12, 12)  # 指定图片像素
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show()

dnn_classifier_vocabulary_file = train_DNN_Classifier_model(
    learning_rate=0.1,
    steps=1000,
    batch_size=25,
    hidden_units=[10, 10],
    feature_columns=[tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_file(key='terms', vocabulary_file='terms.txt'), 2)],
    train_path=train_path,
    test_path=test_path
)

embedding_matrix = dnn_classifier_vocabulary_file.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

for term_index in range(len(informative_terms)):
    embedding_xy = embedding_matrix[term_index]
    plt.text(embedding_xy[0], embedding_xy[1], informative_terms[term_index])

plt.rcParams['figure.figsize'] = (12, 12)
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show()