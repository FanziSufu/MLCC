#!usr/bin/python3
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 打印版本号
print(pd.__version__)

# 创建Series对象
print(pd.Series(['San Francisco', 'San Jose', 'Sacramento']))

# 创建DataFrame对象
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
print(pd.DataFrame({'City name': city_names, 'Population': population}))

# 导入csv数据给DataFrame对象，使用.describe()展示统计数据（中位数，均值，方差等）
california_housing_dataframe = pd.read_csv('california_housing_train.csv', sep=',')
print(california_housing_dataframe.describe())

# 使用.head()返回前n（=5）行的数据
print(california_housing_dataframe.head())

# 使用.hist()画直方图
california_housing_dataframe.hist('housing_median_age')
plt.show()  # 在pycharm中 利用 matplotlib.pyplot.show() 显示图片

# 创建DataFrame对象，并返回 列 的类型和值
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities['City name']))
print(cities['City name'])

# 可以利用索引列表的方法，去索引DataFrame的 列
print(type(cities['City name'][1]))
print(cities['City name'][1])

# 可以使用切片的方法获取DataFrame的 行
print(type(cities[0: 2]))
print(cities[0: 2])

# 可以对Series对象使用Python基本运算指令
print(population / 1000)

# 可以对Series对象使用numpy的科学计算工具
print(np.log(population))

# 使用Series.apply()方法，像映射函数一样，以参数形式接受lambda函数，而该函数会应用于每一个值
print(population.apply(lambda val: val > 1000000))

# 可以通过下面这种方式为DataFrame添加 列
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

# 添加一个布尔值列：城市以圣人命名 且 城市面积大于 50
cities['Named by saint and Area > 50'] = cities['City name'].apply(lambda name: 'San' in name) & \
                                         (cities['Area square miles'] > 50)
print(cities)

# Series和DataFrame对象也定义了index属性，该属性会向每一个 列 或每一个 行 赋一个标识符值。
# 默认情况下，在构造时，pandas会赋反映源数据顺序的索引值。索引值在创建后是稳定的，不会因为数据重新排列而改变
print(city_names.index)
print(cities.index)

# 手动修改DataFrame的行排列顺序
print(cities.reindex([2, 0, 1]))

# 利用numpy的permutation()，重新随机排列DataFrame的行顺序
print(cities.reindex(np.random.permutation(cities.index)))

# 如果reindex的输入数组包含原始索引值中没有的值，reindex会为此类丢失的“索引”添加新的行，并赋值NaN
print(cities.reindex([10, 2, 3, 1, 4, 0]))
