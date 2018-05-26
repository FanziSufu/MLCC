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

    output_targets['median_house_value'] = california_housing_dataframe['median_house_value'] / 1000.0

    return output_targets


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):

    return {tf.feature_column.numeric_column(my_feature) for my_feature in input_features}


def train_dnn_regressor_model(learning_rate, hidden_units, steps, batch_size, feature_columns, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feature_columns, optimizer=my_optimizer)

    training_input_fn = lambda :my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda :my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    print('Training model...')
    print('RMSE (on training data):')
    training_rmses = []
    validation_rmses = []

    for period in range(0, periods):
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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

    print('Final RMSE (on training data): %0.2f' % training_rmse)
    print('Final RMSE (on validation data): %0.2f' % validation_rmse)

    return dnn_regressor


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

dnn_regressor = train_dnn_regressor_model(
    learning_rate=0.005,
    hidden_units=[10, 10],
    steps=2000,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)

california_housing_test_dataframe = pd.read_csv('california_housing_test.csv', sep=',')

test_examples = preprocess_features(california_housing_test_dataframe)
test_targets = preprocess_targets(california_housing_test_dataframe)

predict_test_input_fn = lambda :my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)

test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

test_rmse = math.sqrt(metrics.mean_squared_error(test_targets, test_predictions))

print('Final RMSE (on test data): %0.2f' % test_rmse)
