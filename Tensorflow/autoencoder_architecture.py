

import sys
import tensorflow as tf
print(sys.executable)
print(sys.version)

print('Hey')
tf.reset_default_graph()
input = tf.placeholder(dtype=tf.float32, shape=[64, 224, 224, 3])
conv0 = tf.layers.conv2d(input, filters=32, kernel_size=3, strides=2,
                         padding='same', activation=tf.nn.relu, name='conv0')
print(conv0)
conv1 = tf.layers.conv2d(conv0, filters=64, kernel_size=3, strides=2,
                         padding='same', activation=tf.nn.relu, name='conv1')
print(conv1)
conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=3, strides=1,
                         padding='same', activation=tf.nn.relu, name='conv2')
print(conv2)
conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=3, strides=2,
                         padding='same', activation=tf.nn.relu, name='conv3')
print(conv3)
conv4 = tf.layers.conv2d(conv3, filters=1024, kernel_size=3, strides=2,
                         padding='same', activation=tf.nn.relu, name='conv4')
print(conv4)
conv5 = tf.layers.conv2d(conv4, filters=2048, kernel_size=3, strides=2,
                         padding='same', activation=tf.nn.relu, name='conv5')
print(conv5)
trans1 = tf.layers.conv2d_transpose(
    conv5, filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='trans1')
print(trans1)
trans2 = tf.layers.conv2d_transpose(
    trans1, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='trans2')
print(trans2)
trans3 = tf.layers.conv2d_transpose(
    trans2, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='trans3')
print(trans3)
trans4 = tf.layers.conv2d_transpose(
    trans3, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='trans4')
print(trans4)
trans5 = tf.layers.conv2d_transpose(
    trans4, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='trans5')
print(trans5)
trans6 = tf.layers.conv2d_transpose(
    trans5, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='trans6')
print(trans6)
trans7 = tf.layers.conv2d_transpose(
    trans6, filters=24, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='trans7')
print(trans7)
