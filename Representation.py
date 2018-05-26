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
         'total_rooms',
         'total_bedrooms',
         'population',
         'households',
         'median_income']]

    processed_features = selected_features.copy()
    processed_features['rooms_per_person'] = california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']

    return processed_features


def preprocess_targets(california_housing_dataframe):

    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = california_housing_dataframe['median_house_value'] / 1000.0

    return output_targets


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

correlation_examples = training_examples.copy()
correlation_examples['target'] = training_targets['median_house_value']
correlation_examples.corr()


def construct_feature_columns(input_features):

    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    training_input_fn = lambda :my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda :my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

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

        training_rmse = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_rmse = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))

        training_rmses.append(training_rmse)
        validation_rmses.append(validation_rmse)

        print('Period %02d : %0.2f' % (period, validation_rmse))

    print('Model training finished')

    plt.title('Root Mean Squared Error vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(training_rmses, label='Training')
    plt.plot(validation_rmses, label='Validation')
    plt.legend()
    plt.show()

minimal_features = ['median_income'
                    '', 'latitude']
minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]


def select_and_transform_features(source_df):

    LATITUDE_ARRANGES = zip(range(32, 42), range(33, 43))

    selected_examples = pd.DataFrame()
    selected_examples['median_income'] = source_df['median_income']

    for r in LATITUDE_ARRANGES:
        selected_examples['latitude_%d_to_%d' % r] = source_df['latitude'].apply(lambda l:1.0 if l >= r[0] and l < r[1] else 0.0)

    return selected_examples


selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)

_ = train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)
