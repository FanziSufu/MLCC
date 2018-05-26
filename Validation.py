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
display.display(california_housing_dataframe.describe())


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


training_features = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_features = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# plt.figure(figsize=(13, 8 ))
#
# ax = plt.subplot(1, 2, 1)
# ax.set_title('Validation')
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# plt.scatter(validation_features['longitude'], validation_features['latitude'], cmap='coolwarm',
#             c=validation_targets['median_house_value'] / validation_targets['median_house_value'].max())
#
# ax = plt.subplot(1, 2, 2)
# ax.set_title('Training')
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# plt.scatter(training_features['longitude'], training_features['latitude'], cmap='coolwarm',
#             c = training_targets['median_house_value'] / training_targets['median_house_value'].max())
#
# plt.show()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_features_columns(input_features):

    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def train_model(learning_rate, steps, batch_size, training_features, training_targets, validation_features, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    training_input_fn = lambda : my_input_fn(training_features, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda : my_input_fn(training_features, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda : my_input_fn(validation_features, validation_targets, shuffle=False, num_epochs=1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_features_columns(training_features), optimizer=my_optimizer)

    training_rmses = []
    validation_rmses = []

    print('Traning model...')
    print('RMSE (on training data):')

    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        predictions_training = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions_training = np.array([item['predictions'][0] for item in predictions_training])
        predictions_validation = linear_regressor.predict(input_fn=predict_validation_input_fn)
        predictions_validation = np.array([item['predictions'][0] for item in predictions_validation])

        training_rmse = math.sqrt(metrics.mean_squared_error(predictions_training, training_targets))
        validation_rmse = math.sqrt(metrics.mean_squared_error(predictions_validation, validation_targets))
        training_rmses.append(training_rmse)
        validation_rmses.append(validation_rmse)

        print('Period %02d : %0.2f' % (period, training_rmse))

    print('Model training finished.')
    print('Final RMSE: %0.2f' % training_rmse)

    plt.title('RMSE vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(training_rmses, label='Training')
    plt.plot(validation_rmses, label='Validation')
    plt.legend()
    plt.show()

    return linear_regressor

linear_regressor = train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=20,
    training_features=training_features,
    training_targets=training_targets,
    validation_features=validation_features,
    validation_targets=validation_targets
)

california_housing_test_dataframe = pd.read_csv('california_housing_test.csv', sep=',')

test_features = preprocess_features(california_housing_test_dataframe)
test_targets = preprocess_targets(california_housing_test_dataframe)

test_input_fn = lambda : my_input_fn(test_features, test_targets, shuffle=False, num_epochs=1)

predictions_test = linear_regressor.predict(input_fn=test_input_fn)
predictions_test = np.array([item['predictions'][0] for item in predictions_test])

test_rmse = math.sqrt(metrics.mean_squared_error(predictions_test, test_targets))

print('tese_rmse: %0.2f' % test_rmse)
