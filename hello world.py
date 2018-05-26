#!usr/bin/python3
# -*- coding: UTF-8 -*-


import tensorflow as tf


c = tf.constant('Hello, world!')
tf.assign()

with tf.Session() as sess:
    print(sess.run(c))
