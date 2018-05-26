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
         'total_rooms',
         'total_bedrooms',
         'population',
         'households',
         'median_income',
         'housing_median_age']]

    processed_features = selected_features.copy()

    processed_features['rooms_per_person'] = california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']

    return processed_features


def preprocess_targets(california_housing_dataframe):

    output_targets = pd.DataFrame()

    output_targets['median_house_value_is_high'] = (california_housing_dataframe['median_house_value'] > 265000).astype(float)

    return output_targets


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def get_quantile_based_boundaries(feature_values, num_buckets):

    boundaries = np.arange(1, num_buckets + 1) / (num_buckets + 1)

    quantiles = feature_values.quantile(boundaries)

    return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns():

    bucketized_longitude = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'), get_quantile_based_boundaries(training_examples['longitude'], 50))
    bucketized_latitude = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), get_quantile_based_boundaries(training_examples['latitude'], 50))
    bucketized_total_rooms = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('total_rooms'), get_quantile_based_boundaries(training_examples['total_rooms'], 10))
    bucketized_total_bedrooms = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('total_bedrooms'), get_quantile_based_boundaries(training_examples['total_bedrooms'], 10))
    bucketized_population = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('population'), get_quantile_based_boundaries(training_examples['population'], 10))
    bucketized_households = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('households'), get_quantile_based_boundaries(training_examples['households'], 10))
    bucketized_median_income = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('median_income'), get_quantile_based_boundaries(training_examples['median_income'], 10))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('housing_median_age'), get_quantile_based_boundaries(training_examples['housing_median_age'], 10))
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('rooms_per_person'), get_quantile_based_boundaries(training_examples['rooms_per_person'], 10))

    long_x_lati = tf.feature_column.crossed_column({bucketized_longitude, bucketized_latitude}, hash_bucket_size=1000)

    return {bucketized_longitude, bucketized_latitude, bucketized_total_rooms, bucketized_total_bedrooms, bucketized_population, bucketized_households, bucketized_median_income,
            bucketized_housing_median_age, bucketized_rooms_per_person, long_x_lati}


def model_size(estimator):

    variables = estimator.get_variable_names()

    size = 0

    for variable in variables:
        if not any(x in variable for x in ['global_steps', 'centered_bias_weight', 'bias_weight', 'Ftrl']):
            size += np.count_nonzero(estimator.get_variable_value(variable))

    return size


def train_linear_classifier_model(learning_rate, regularization_strength, steps, batch_size, feature_columns, training_examples, training_targets, validation_examples, validation_targets):

    periods = 7
    steps_per_period = steps / periods

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=my_optimizer)

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

    print('Model training finished')

    plt.title('Logless vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('Logless')
    plt.tight_layout()
    plt.plot(training_log_lesses, label='Training')
    plt.plot(validation_log_lesses, label='Validation')
    plt.legend()
    plt.show()

    return linear_classifier


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

linear_classifier = train_linear_classifier_model(
    learning_rate=0.1,
    regularization_strength=0.1,
    steps=300,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

print('Model size:', model_size(linear_classifier))
