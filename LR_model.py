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
california_housing_dataframe['median_house_value'] /= 1000.0


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature='total_rooms'):

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = 'median_house_value'
    targets = california_housing_dataframe[my_label]
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    training_input_fn = lambda : my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda : my_input_fn(my_feature_data, targets, shuffle=False, num_epochs=1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title('Learned line by Period')
    plt.xlabel(my_feature)
    plt.ylabel(my_label)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    print('Training model...')
    print('RMSE (on training data):')
    root_mean_squared_errors = []

    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print('period %02d : %0.2f' % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)

        y_extents = np.array([0, sample[my_label].max()])
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()), sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])

    print('Model training finished')

    plt.subplot(1, 2, 2)
    plt.title('Root Mean Squared Error vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    calibration_data = pd.DataFrame()
    calibration_data['predictions'] = pd.Series(predictions)
    calibration_data['targets'] = pd.Series(targets)
    display.display(calibration_data.describe())

    print('Final RMSE (on training data): %0.2f' % root_mean_squared_error)

train_model(learning_rate=0.00002, steps=500, batch_size=5)
plt.show()