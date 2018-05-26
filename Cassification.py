#!usr/bin/python3
# -*- coding: UTF-8 -*-

import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset
import numpy as np
import pandas as pd
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv('california_housing_train.csv', sep=',')
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):

    selected_features = california_housing_dataframe[
        ['longitude',
         'latitude',
         'housing_median_age',
         'population',
         'total_rooms',
         'total_bedrooms',
         'median_income',
         'households']]

    processed_features = selected_features.copy()

    processed_features['rooms_per_person'] = california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']

    return processed_features


def preprocess_targets(california_housing_dataframe):

    output_targets = pd.DataFrame()

    output_targets['median_house_value_is_high'] = (california_housing_dataframe['median_house_value'] > 265000).astype(float)

    return output_targets


def construct_feature_columns(input_features):

    return {tf.feature_column.numeric_column(my_feature) for my_feature in input_features}


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_linear_regressor_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

    training_input_fn = lambda :my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda :my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    print('Training model...')
    print('RMSE (on training data):')
    training_rmses = []
    validation_rmses = []

    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_rmse = math.sqrt(metrics.mean_squared_error(training_targets, training_predictions))
        validation_rmse = math.sqrt(metrics.mean_squared_error(validation_targets, validation_predictions))

        training_rmses.append(training_rmse)
        validation_rmses.append(validation_rmse)

        print('Period %02d : %0.2f' % (period, training_rmse))

    print('Model training finished')

    plt.title('Root Mean Squared Error vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(training_rmses, label='Training')
    plt.plot(validation_rmses, label='Validation')
    plt.legend()
    plt.show()

    return linear_regressor


def train_linear_classifier_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

    training_input_fn = lambda :my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda :my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    training_log_lesses = []
    validation_log_lesses = []

    print('Training model...')
    print('Logless (on training data):')

    for period in range(0, periods):
        linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)

        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_less = metrics.log_loss(training_targets, training_probabilities)
        validation_log_less = metrics.log_loss(validation_targets, validation_probabilities)

        training_log_lesses.append(training_log_less)
        validation_log_lesses.append(validation_log_less)

        print('Peirod %02d : %0.2f' % (period, training_log_less))

    print('Model tarining finished')

    plt.title('Logless vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('Logless')
    plt.tight_layout()
    plt.plot(training_log_lesses, label='Training')
    plt.plot(validation_log_lesses, label='Validation')
    plt.show()

    return linear_classifier


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print('AUC on the validation set: %0.2f' % evaluation_metrics['auc'])
print('Accuracy on the validation set: %0.2f' % evaluation_metrics['accuracy'])

validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(validation_targets, validation_probabilities)

plt.plot(false_positive_rate, true_positive_rate, label='our model')
plt.plot([0, 1], [0, 1], label='random classifier')
_ = plt.legend(loc=2)
