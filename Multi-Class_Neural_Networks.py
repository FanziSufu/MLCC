#!usr/bin/python3
# -*- coding: UTF-8 -*-

import glob
import io
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(io.open('mnist_train_small.csv', 'r'), sep=',', header=None)  # 使用io.open而不直接使用路径，是为了以只读模式打开本地文件； 因为文件不含列名称，所以设定hearder=None,以自动生成数字列名称
mnist_dataframe = mnist_dataframe.head(10000)
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
print(mnist_dataframe.head())


def parse_labels_and_features(dataset):

    labels = dataset[0]

    features = dataset.loc[:, 1: 784]  # 获取第1列到第784列所有行的数据。这里的1和784是索引名称
    features = features / 255  # 把特征值缩放到[0, 1]

    return labels, features


training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])

# rand_example = np.random.choice(training_examples.index)  # 获取一个随机样本的索引
# _, ax = plt.subplots()  # 创建一个图
# ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))  # 根据索引行的值，转换为28*28的二位矩阵，画出矩阵
# ax.set_title('Label: %i' % training_targets.loc[rand_example])
# ax.grid(False)  # 隐藏网格


def construct_feature_columns():

    return {tf.feature_column.numeric_column('pixels', shape=784)}  # 这里不像以前那样使用列名称遍历的方式，是因为这个数据集的列名称是数字，而非str。 shape指定了这个关键字的大小


def create_training_input_fn(features, labels, batch_size, shuffle=True, num_epochs=None):

    def _input_fn(shuffle=True, num_epochs=None):
        idx = np.random.permutation(features.index)
        raw_features = {'pixels': features.reindex(idx).values}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):

    def _input_fn():
        raw_features = {'pixels': features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def train_linear_classification_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(), n_classes=10, optimizer=my_optimizer, config=tf.estimator.RunConfig(keep_checkpoint_max=1))

    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)

    print('Training model...')
    print('Logless (on validation data):')
    training_loglesses = []
    validation_loglesses = []

    for period in range(0, periods):
        linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = list(linear_classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])  # 获取预测为各类别的概率
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])  # 获取预测值
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)  # 将类别转化为独热编码，以便使用metrics.log_loss函数

        validation_predictions = list(linear_classifier.predict(input_fn=predict_validation_input_fn))
        validaiton_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_logless = metrics.log_loss(training_targets, training_pred_one_hot)  # 这里可以使用另一种预测值， validation_probabilities 来计算损失函数。 但损失结果的数值，小了10倍。 而且不够准确。
        validation_logless = metrics.log_loss(validation_targets, validation_pred_one_hot)

        training_loglesses.append(training_logless)
        validation_loglesses.append(validation_logless)

        print('Period %2d : %0.2f' % (period, validation_logless))

    print('Model training finished')

    _ = map(os.remove, glob.glob(os.path.join(linear_classifier.model_dir, 'events.out.tfevents*')))  # 删除event文件，释放存储空间

    final_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)  # 计算准确度
    print('Final Accuracy (on validations data): %0.2f' % accuracy)

    plt.title('Logless vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('Logless')
    plt.tight_layout()
    plt.plot(training_loglesses, label='Training')
    plt.plot(validation_loglesses, label='Validation')
    plt.legend()
    plt.show()

    cm = metrics.confusion_matrix(validation_targets, final_predictions)  # 创建混淆矩阵
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  # 标准化混淆矩阵，使每一数值除以所在行的数值和。np.newaxis = None
    ax = sns.heatmap(cm_normalized, cmap='bone_r')  # 用混淆矩阵创建热图，方便观察结果
    ax.set_aspect(1)  # 设置纵横比为1:1
    plt.title('Confusion matrix')
    plt.ylabel('True labels')
    plt.xlabel('Predict labels')
    plt.show()

    return linear_classifier


# _ = train_linear_classification_model(
#     learning_rate=0.03,
#     steps=1000,
#     batch_size=30,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)


def train_DNN_Classifier_model(learning_rate, steps, batch_size, hidden_units, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=construct_feature_columns(), n_classes=10, optimizer=my_optimizer)

    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)

    print('Training model...')
    print('Logless (on validation data):')
    training_loglesses = []
    validation_loglesses = []

    for period in range(0, periods):
        dnn_classifier.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = dnn_classifier.predict(input_fn=predict_training_input_fn)
        training_pred_class_ids = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_ids, 10)

        validation_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
        validation_pred_class_ids = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_ids, 10)

        training_logless = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_logless = metrics.log_loss(validation_targets, validation_pred_one_hot)

        training_loglesses.append(training_logless)
        validation_loglesses.append(validation_logless)

        print('Period %02d : %0.2f' % (period, validation_logless))

    print('Model training finished')

    _ = map(os.remove, glob.glob(os.path.join(dnn_classifier.model_dir, 'events.out.tfevents*')))

    plt.title('Logless vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('Logless')
    plt.tight_layout()
    plt.plot(training_loglesses, label='Training')
    plt.plot(validation_loglesses, label='Validation')
    plt.legend()
    plt.show()

    final_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print('Accuracy (on validation data): %0.2f' % accuracy)

    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:]
    ax = sns.heatmap(cm_normalized, cmap='bone_r')
    ax.set_aspect(1)
    plt.title('Confusion Matrix')
    plt.xlabel('Predict labels')
    plt.ylabel('True labels')
    plt.show()

    return dnn_classifier

dnn_classifier = train_DNN_Classifier_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


mnist_test_dataframe = pd.read_csv(io.open('mnist_test.csv', 'r'), sep=',', header=None)
test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)

predict_test_input_fn = create_predict_input_fn(test_examples, test_targets, batch_size=100)
test_predictions = dnn_classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['class_ids'][0] for item in test_predictions])
accuracy = metrics.accuracy_score(test_targets, test_predictions)

print('Accuracy (on test data): %0.2f' % accuracy)

print(dnn_classifier.get_variable_names())

weights0 = dnn_classifier.get_variable_value('dnn/hiddenlayer_0/kernel')
print('weight0 shape:', weights0.shape)

num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
