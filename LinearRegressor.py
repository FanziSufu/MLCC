#!usr/bin/python3
# -*- coding: UTF-8 -*-

import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


'''tf.logging--记录日志
TensorFlow用五个不同级别的日志信息。为了升序的严重性，他们是调试DEBUG，信息INFO，警告WARN，
错误ERROR和致命FATAL的。
当你配置日志记录在任何级别，TensorFlow将输出对应于更高程度的严重性和所有级别的日志信息。
例如，如果设置错误的日志记录级别，将得到包含错误和致命消息的日志输出，并且如果设置了调试级别，
则将从所有五个级别获取日志消息。
默认情况下，TensorFlow配置在日志记录级别的WARN，但当跟踪模型的训练，你会想要调整水平到INFO，
这将提供额外的反馈如进程中的fit操作。
'''
tf.logging.set_verbosity(tf.logging.ERROR)  # 设置ERROR级别的日志，这样只会记录ERROR和FATAL的信息
pd.options.display.max_rows = 10  # 设置显示的最大行数为10，超过则使用省略号
pd.options.display.float_format = '{:.1f}'.format  # 设置浮点数的格式

california_housing_dataframe = pd.read_csv('california_housing_train.csv', sep=',')  # 加载数据
california_housing_dataframe = california_housing_dataframe.reindex\
    (np.random.permutation(california_housing_dataframe.index))  # 随机排列初始数据
california_housing_dataframe['median_house_value'] /= 1000.0  # 将房价调整为以 千 为单位
print(california_housing_dataframe)  # 显示数据
print(california_housing_dataframe.describe())  # 显示数据的统计信息

# 获取total_rooms列的数据，定义其为输入特征。注意使用2个方括号，这样可以获取其列名
my_feature = california_housing_dataframe[['total_rooms']]
# 为total_rooms创建特征列，用来表示特征的数据类型（分类数据或数值数据）
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# 定义标签（即预测目标）。这里只使用一个方括号
targets = california_housing_dataframe['median_house_value']

# 建立梯度下降优化模型，用来训练模型
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 为了安全起见，通过clip_gradients_by_norm将梯度裁剪应用到我们的优化器。
# 梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 配置线性回归模型
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                                optimizer=my_optimizer)


# 定义一个输入函数（只有一个特征），告诉TensorFlow如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # 把pandas特征数据转换成NumPy数组字典
    features = {key:np.array(value) for key,value in dict(features).items()}

    # 构建数据集
    ds = Dataset.from_tensor_slices((features, targets))
    # 配置单步的样本数量,每批次传递给模型处理的数据量为batch_size；如果将默认值num_epochs=None传递到repeat()，输入数据会无限重复下去，而没有次数限制
    ds = ds.batch(batch_size).repeat(num_epochs)

    # 打乱数据，缓冲大小为10000，以便数据在训练期间以随机方式传递到模型
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # 为数据集构建一个迭代器，并向LinearRegressor返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 在linear_regressor上调用trani()来训练模型。将my_input_fn封装在lambda中，以便将my_feature和targets作为参数传入
_ = linear_regressor.train(input_fn = lambda: my_input_fn(my_feature, targets), steps=100)

# 设置num_epochs=1,不重复数据，设置shuffle=False，表示传入所有17000个样本数据（这里如果设置num_epochs=2，则会传入34000个数据）
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# 调用predict()，获取预测值
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# 把预测值的格式改成Numpy数据，以便计算错误指标
predictions = np.array([item['predictions'][0] for item in predictions])

# 打印均方误差和均方根误差，这里调用了sklearn的metrics函数
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print('Mean Squared Error (on training data): %0.3f' % mean_squared_error)
print('Root Mean Squared Error (on training data): %0.3f' % root_mean_squared_error)

# 打印源数据的最大值，最小值，其差，与均值方差进行比较
min_house_value = california_housing_dataframe['median_house_value'].min()
max_house_value = california_housing_dataframe['median_house_value'].max()
min_max_difference = max_house_value - min_house_value

print('Min. Median House Value: %0.3f' % min_house_value)
print('Max. Median House Value: %0.3f' % max_house_value)
print('Difference beiween Min. and Max.: %0.3f' % min_max_difference)
print('Root Mean Squared Error (on training data): %0.3f' % root_mean_squared_error)

# 构建DataFrame对象，利用describe()比较统计数据
california_data = pd.DataFrame()
california_data['predictions'] = pd.Series(predictions)
california_data['targets'] = pd.Series(targets)
print(california_data.describe())

# 随机选出300个训练样本
sample = california_housing_dataframe.sample(n=300)

# 获取随机样本total_rooms的最小值、最大值
x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()

# 获取训练后得到的权重weight和偏差想bias
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# 计算y值，用于绘制训练得到的图线
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# 绘制训练得到的图线
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# 设置纵坐标和横坐标的名称
plt.ylabel('median_house_value')
plt.xlabel('total_rooms')

# 绘制样本的散点图
plt.scatter(sample['total_rooms'], sample['median_house_value'])

# 展现图片，观察训练得到的图线对样本散点图的拟合情况
plt.show()

# 定义一个训练模型函数，包含上面的步骤，方便调试超参数
def train_model(learning_rate, steps, batch_size, input_feature='total_rooms'):

    # 设置反馈区间periods，共计10次，输出损失值
    periods = 10
    steps_per_period = steps / periods

    # 定义特征并配置特征列
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = 'median_house_value'
    targets = california_housing_dataframe[my_label]
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # 配置训练用输入函数和预测用输入函数
    training_input_fn = lambda : my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda : my_input_fn(my_feature_data, targets, batch_size=1, shuffle=False)

    # 配置线性回归模型
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    # 画出样本散点图
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title('Learned Line by Period')
    plt.xlabel(my_feature)
    plt.ylabel(my_label)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]  # 创建不同的颜色值

    # 打印信息
    print('Training Model...')
    print('RMSE (on training data):')
    root_mean_squared_errors = []

    # 循环训练模型
    for period in range(0, periods):
        # 按period训练模型
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        # 获取预测值，并转换成np数组格式
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # 计算均值跟方差RMSE
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print('period: %02d : %0.2f' % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)

        # 选择两个合适的点（从4个点选出中间2个点）画出每一period获得的模型图
        y_extents = np.array([0, sample[my_label].max()])
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()), sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])

    print('Model training finished')

    # 画出均值跟方差变化曲线
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # 比较预测值和实际值的统计信息
    california_data = pd.DataFrame()
    california_data['predictions'] = pd.Series(predictions)
    california_data['targets'] = pd.Series(targets)
    display.display(california_data.describe())

    print('Final RMSE (on training data): %0.2f' % root_mean_squared_error)
