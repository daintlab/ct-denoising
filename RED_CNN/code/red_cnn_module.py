# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018
@author: yeohyeongyu
"""

import tensorflow.contrib.layers as tcl
import tensorflow as tf


def redcnn(in_image, name="redcnn", reuse = True, kernel_size = [5,5], filter_size = 96, conv_stride = 1, initial_std = 0.01):
    
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        """
        encoder
        """
        #conv layer1
        conv1 = tcl.conv2d(in_image,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std), \
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        conv1 = tf.nn.relu(conv1)
        #conv layer2
        conv2 = tcl.conv2d(conv1,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std), \
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        conv2 = shortcut_deconv8 = tf.nn.relu(conv2)
        #conv layer3
        conv3 =  tcl.conv2d(conv2,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std),\
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        conv3 = tf.nn.relu(conv3)
        #conv layer4
        conv4 = tcl.conv2d(conv3,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std),\
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        conv4 = shortcut_deconv6 = tf.nn.relu(conv4)
        #conv layer5
        conv5 = tcl.conv2d(conv4,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std), \
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        conv5 = tf.nn.relu(conv5)

        """
        decoder
        """
        #deconv 6 + shortcut (residual style)
        deconv6 = tcl.conv2d_transpose(conv5,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std),\
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        deconv6 +=  shortcut_deconv6
        deconv6 = tf.nn.relu(deconv6)
        #deconv 7
        deconv7 = tcl.conv2d_transpose(deconv6,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std), \
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        deconv7 = tf.nn.relu(deconv7)
        #deconv 8 + shortcut 
        deconv8 = tcl.conv2d_transpose(deconv7, filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std),\
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        deconv8 +=  shortcut_deconv8
        deconv8 = tf.nn.relu(deconv8)
        #deconv 9
        deconv9 = tcl.conv2d_transpose(deconv8,  filter_size, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std),\
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        deconv9 = tf.nn.relu(deconv9)
        #deconv 10 + shortcut 
        deconv10 = tcl.conv2d_transpose(deconv9, 1, kernel_size, conv_stride, padding='valid', weights_initializer = tf.random_normal_initializer(stddev=initial_std), \
                           biases_initializer= tf.zeros_initializer(), activation_fn  = None)
        deconv10 += in_image
        output = tf.nn.relu(deconv10)

        return output
