#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-07-27 13:14
'''

from numpy import *
import os
import tensorflow as tf

weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
bias_initializer = tf.constant_initializer(0.1)
def variable(name, shape, initializer, dtype=tf.float32, regularizer=None, trainable=True):
    with tf.device('/cpu:0'):
        return tf.get_variable(name, shape=shape, dtype=dtype,
                               initializer=initializer, regularizer=regularizer,
                               trainable=trainable)

def alex_l2_regularizer(weight=0.0005, scope=None):
  """Define a L2 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.name_scope(scope, 'L2Regularizer', [tensor]):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer

#def xwplusb(x, w, b) :

def relu(input):
      ts = tf.Variable(input, name="W")      
      ret = tf.nn.relu(ts)
      return ret
    
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        #input_groups = tf.split(3, group, input)
        #kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        #conv = tf.concat(3, output_groups)
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

#def get6W() :
#    with tf.variable_scope('alexnet_v1',reuse=True) as sc:
#        l2_regularizer = alex_l2_regularizer(0.0005)                
#        fc6W = variable('fc6_w', [9216, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
#        return fc6W
def softmax(inputs):
    x = tf.Variable(inputs)    
    ret = tf.nn.softmax(inputs)
    return ret;
def alexnet_v1(inputs, num_classes, is_training=True, weight_decay=0.0005, scope=None):
    with tf.variable_scope(scope, 'alexnet_v1', [inputs]) as sc:
        l2_regularizer = alex_l2_regularizer(weight_decay)
        conv1W = variable('conv1_w', [11, 11, 3, 96], initializer=weights_initializer, regularizer=l2_regularizer)
        conv1b = variable('conv1_b', [96], initializer=bias_initializer)
        # Conv1 - Relu - Lrn - Maxpool
        conv1 = tf.nn.relu(conv(inputs, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=2,
                                                  alpha=2e-05,
                                                  beta=0.75,
                                                  bias=1.0)
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Conv2 - Relu - Lrn - Maxpool
        conv2W = variable('conv2_w', [5, 5, 48, 256], initializer=weights_initializer, regularizer=l2_regularizer)
        conv2b = variable('conv2_b', [256], initializer=bias_initializer)
        conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=2,
                                                  alpha=2e-05,
                                                  beta=0.75,
                                                  bias=1.0)
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Conv3 - Conv4 - Conv5 - Maxpool5 - Fc6 -Fc7
        conv3W = variable('conv3_w', [3, 3, 256, 384], initializer=weights_initializer, regularizer=l2_regularizer)
        conv3b = variable('conv3_b', [384], initializer=bias_initializer)
        conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
        conv4W = variable('conv4_w', [3, 3, 192, 384], initializer=weights_initializer, regularizer=l2_regularizer)
        conv4b = variable('conv4_b', [384], initializer=bias_initializer)
        conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
        conv5W = variable('conv5_w', [3, 3, 192, 256], initializer=weights_initializer, regularizer=l2_regularizer)
        conv5b = variable('conv5_b', [256], initializer=bias_initializer)
        conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc6x = tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])
#        fc6W = variable('fc6_w', [9216, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
#        sc.reuse_variable()
                
        return fc6x, dict()
#        fc6W = variable('fc6_w', [9216, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
#        fc6b = variable('fc6_b', [4096], initializer=bias_initializer)
#        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
#        fc7W = variable('fc7_w', [4096, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
#        fc7b = variable('fc7_b', [4096], initializer=bias_initializer)
#        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
#        fc8W = variable('fc8_w', [4096, 1000], initializer=weights_initializer, regularizer=l2_regularizer)
#        fc8b = variable('fc8_b', [1000], initializer=bias_initializer)
#        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
#        return fc6W, fc6b, tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]) , dict()#, dict()#,tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]),fc6W, fc6b
#        prob = tf.nn.softmax(fc8)
#        print('prob: %s' % prob)
#        return prob,dict(), tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])


"""def alexnet_v2(inputs, num_classes, is_training=True, weight_decay=0.0005, scope=None):
    with tf.variable_scope(scope, 'alexnet_v1', [inputs]) as sc:
        l2_regularizer = alex_l2_regularizer(weight_decay)
        conv1W = variable('conv1_w', [11, 11, 3, 96], initializer=weights_initializer, regularizer=l2_regularizer)
        conv1b = variable('conv1_b', [96], initializer=bias_initializer)
        # Conv1 - Relu - Lrn - Maxpool
        conv1 = tf.nn.relu(conv(inputs, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=2,
                                                  alpha=2e-05,
                                                  beta=0.75,
                                                  bias=1.0)
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Conv2 - Relu - Lrn - Maxpool
        conv2W = variable('conv2_w', [5, 5, 48, 256], initializer=weights_initializer, regularizer=l2_regularizer)
        conv2b = variable('conv2_b', [256], initializer=bias_initializer)
        conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=2,
                                                  alpha=2e-05,
                                                  beta=0.75,
                                                  bias=1.0)
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Conv3 - Conv4 - Conv5 - Maxpool5 - Fc6 -Fc7
        conv3W = variable('conv3_w', [3, 3, 256, 384], initializer=weights_initializer, regularizer=l2_regularizer)
        conv3b = variable('conv3_b', [384], initializer=bias_initializer)
        conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
        conv4W = variable('conv4_w', [3, 3, 192, 384], initializer=weights_initializer, regularizer=l2_regularizer)
        conv4b = variable('conv4_b', [384], initializer=bias_initializer)
        conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
        conv5W = variable('conv5_w', [3, 3, 192, 256], initializer=weights_initializer, regularizer=l2_regularizer)
        conv5b = variable('conv5_b', [256], initializer=bias_initializer)
        conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc6W = variable('fc6_w', [9216, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
        fc6b = variable('fc6_b', [4096], initializer=bias_initializer)
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        fc7W = variable('fc7_w', [4096, 4096], initializer=weights_initializer, regularizer=l2_regularizer)
        fc7b = variable('fc7_b', [4096], initializer=bias_initializer)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        fc8W = variable('fc8_w', [4096, 1000], initializer=weights_initializer, regularizer=l2_regularizer)
        fc8b = variable('fc8_b', [1000], initializer=bias_initializer)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        prob = tf.nn.softmax(fc8)
        return prob,dict()
"""
