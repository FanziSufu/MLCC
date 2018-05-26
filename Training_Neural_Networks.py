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


def construct_feature_columns(input_features):

    return {tf.feature_column.numeric_column(my_feature) for my_feature in input_features}


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features,labels


def train_dnn_regressor_model(my_optimizer, steps, batch_size, hidden_units, training_examples, training_targets, validation_examples, validation_targets):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer, hidden_units=hidden_units)

    training_input_fn = lambda :my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda :my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda :my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    print('Trining model...')
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

        print('Peirod %02d : %0.2f' % (period, training_rmse))

    print('Model training finished')

    plt.title('RMSE vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(training_rmses, label='Training')
    plt.plot(validation_rmses, label='Validation')
    plt.legend()
    plt.show()

    return dnn_regressor, training_rmses, validation_rmses


def linear_normalize(series):

    series_max = series.max()
    series_min = series.min()
    scare = (series_max - series_min) / 2.0

    return series.apply(lambda x:(x - series_min) / scare - 1.0)


def log_normalize(series):

    return series.apply(lambda x:math.log(x + 1.0))


def clip_normalize(series, clip_min, clip_max):

    return series.apply(lambda x:(min(max(x, clip_min), clip_max)))


def z_score_normalize(series):

    mean = series.mean()
    std = series.std()

    return series.apply(lambda x:(x - mean) / std)


def binary_normalize(series, thresholds):

    return series.apply(lambda x:1.0 if x > thresholds else 0.0)


def normalize(examples):

    processed_features = pd.DataFrame()

    processed_features['housing_median_age'] = linear_normalize(examples['housing_median_age'])
    processed_features['longitude'] = linear_normalize(examples['longitude'])
    processed_features['latitude'] = linear_normalize(examples['latitude'])

    processed_features['households'] = log_normalize(examples['households'])
    processed_features['median_income'] = log_normalize(examples['median_income'])
    processed_features['total_bedrooms'] = log_normalize(examples['total_bedrooms'])

    processed_features['total_rooms'] = clip_normalize(examples['total_rooms'], 0, 10000)
    processed_features['poputation'] = clip_normalize(examples['population'], 0, 5000)
    processed_features['rooms_per_person'] = clip_normalize(examples['rooms_per_person'], 0, 5)

    return processed_features



training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targetrs = preprocess_targets(california_housing_dataframe.tail(5000))

_ = training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=2)

normalized_training_examples = normalize(training_examples)
normalized_validation_examples = normalize(validation_examples)


_, adagrad_training_lesses, adagrad_validation_lesses = train_dnn_regressor_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targetrs)


_, adam_training_lesses, adam_validation_lesses = train_dnn_regressor_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targetrs)


plt.title('Adagran vs. Adam')
plt.xlabel('Periods')
plt.ylabel('RMSE')
plt.tight_layout()
plt.plot(adagrad_training_lesses, label='Adagrad_Training')
plt.plot(adagrad_validation_lesses, label='Adagrad_Validation')
plt.plot(adam_training_lesses, label='Adam_Training')
plt.plot(adam_validation_lesses, label='Adam_Validation')
plt.legend()
plt.show()
