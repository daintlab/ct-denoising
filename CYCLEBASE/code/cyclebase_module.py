# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:52:02 2018

@author: yeohyeongyu
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import copy

def discriminator(args, image, reuse=False, name='discriminator'):
    if args.strct == 'ident':
        def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
            with tf.variable_scope(name):
                padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                return tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, \
                    strides=s, padding="valid", \
                    kernel_initializer=tf.random_normal_initializer(0, 0.02))
        def first_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
            with tf.variable_scope(name):
                return lrelu(conv2d(input_, out_channels, ks=ks, s=s))
        def conv_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
            with tf.variable_scope(name):
                return lrelu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))
        def last_layer(input_, out_channels, ks=4, s=1, name='disc_conv_layer'):
            with tf.variable_scope(name):
                return tf.contrib.layers.fully_connected(conv2d(input_, out_channels, ks=ks, s=s), out_channels)

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            l1 = first_layer(image, args.ndf, ks=4, s=2, name='disc_layer1')
            l2 = conv_layer(l1, args.ndf*2, ks=4, s=2, name='disc_layer2')
            l3 = conv_layer(l2, args.ndf*4, ks=4, s=2, name='disc_layer3')
            l4 = conv_layer(l3, args.ndf*8, ks=4, s=1, name='disc_layer4')
            l5 = last_layer(l4, args.img_channel, ks=4, s=1, name='disc_layer5')
            return l5

    elif  args.strct == 'cyc':
        def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
            with tf.variable_scope(name):
                return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(image, args.ndf, name='d_h0_conv'))
            h1 = lrelu(instance_norm(conv2d(h0, args.ndf*2, name='d_h1_conv'), 'd_bn1'))
            h2 = lrelu(instance_norm(conv2d(h1, args.ndf*4, name='d_h2_conv'), 'd_bn2'))
            h3 = lrelu(instance_norm(conv2d(h2, args.ndf*8, s=1, name='d_h3_conv'), 'd_bn3'))
            h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
            return h4
        
        
def generator(args, image, reuse=False, name="generator"):
    if args.strct == 'ident':
        def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
            with tf.variable_scope(name):
                padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                return tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, \
                    strides=s, padding="valid", \
                    kernel_initializer=tf.random_normal_initializer(0, 0.02))
        def conv_layer(input_, out_channels, ks=3, s=1, name='conv_layer'):
            with tf.variable_scope(name):
                return tf.nn.relu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))       
        def gen_module(input_,  out_channels, ks=3, s=1, name='gen_module'):
            with tf.variable_scope(name):
                ml1 = conv_layer(input_, out_channels, ks, s, name=name + '_l1')
                ml2 = conv_layer(ml1, out_channels, ks, s, name=name + '_l2')
                ml3 = conv_layer(ml2, out_channels, ks, s, name=name + '_l3')
                concat_l = input_ + ml3
                m_out = tf.nn.relu(concat_l)
                return m_out
            
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            l1 = conv_layer(image, args.ngf, name='convlayer1') 
            module1 = gen_module(l1, args.ngf, name='gen_module1')
            module2 = gen_module(module1, args.ngf, name='gen_module2')
            module3 = gen_module(module2, args.ngf, name='gen_module3')
            module4 = gen_module(module3, args.ngf, name='gen_module4')
            module5 = gen_module(module4, args.ngf, name='gen_module5')
            module6 = gen_module(module5, args.ngf, name='gen_module6')
            concate_layer = tf.concat([l1, module1, \
                    module2, module3, module4, module5, module6], axis=3, name='concat_layer')
            concat_conv_l1 = conv_layer(concate_layer, args.ngf, ks=3, s=1, name='concat_convlayer1')
            last_conv_layer = conv_layer(concat_conv_l1, args.nglf, ks=3, s=1, name='last_conv_layer')
            if args.norm == 'n-11':
                output= tf.math.tanh(tf.add(conv2d(last_conv_layer, \
                   args.img_channel, ks=3, s=1), image, name = 'output'))
            else:
                output= tf.math.sigmoid(tf.add(conv2d(last_conv_layer, \
                   args.img_channel, ks=3, s=1), image, name = 'output'))
            return output 

    elif  args.strct == 'cyc':
        def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
            with tf.variable_scope(name):
                return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)
        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x
        
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            
            
            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = tf.nn.relu(instance_norm(conv2d(c0, args.ngf, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
            c2 = tf.nn.relu(instance_norm(conv2d(c1, args.ngf*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            c3 = tf.nn.relu(instance_norm(conv2d(c2, args.ngf*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
            # define G network with 9 resnet blocks
            r1 = residule_block(c3, args.ngf*4, name='g_r1')
            r2 = residule_block(r1, args.ngf*4, name='g_r2')
            r3 = residule_block(r2, args.ngf*4, name='g_r3')
            r4 = residule_block(r3, args.ngf*4, name='g_r4')
            r5 = residule_block(r4, args.ngf*4, name='g_r5')
            r6 = residule_block(r5, args.ngf*4, name='g_r6')
            r7 = residule_block(r6, args.ngf*4, name='g_r7')
            r8 = residule_block(r7, args.ngf*4, name='g_r8')
            r9 = residule_block(r8, args.ngf*4, name='g_r9')

            d1 = deconv2d(r9, args.ngf*2, 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
            d2 = deconv2d(d1, args.ngf, 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            if args.norm == 'n-11':
                pred = tf.math.tanh(conv2d(d2, args.img_channel, 7, 1, padding='VALID', name='g_pred_c'))
            else:
                pred = tf.math.sigmoid(conv2d(d2, args.img_channel, 7, 1, padding='VALID', name='g_pred_c'))

            return pred

        
#### network components
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x)
    
def batchnorm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(input_, axis=3, epsilon=1e-5, \
            momentum=0.1, training=True, \
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    
def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset   

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)
       
#### loss
def least_square(A, B):
    return tf.reduce_mean((A - B)**2)
    
def cycle_loss(A, F_GA, B, G_FB, lambda_):
    return lambda_ * (tf.reduce_mean(tf.abs(A - F_GA)) \
       + tf.reduce_mean(tf.abs(B - G_FB)))

def identity_loss(A, G_B, B, F_A, gamma):
    return   gamma * (tf.reduce_mean(tf.abs(G_B - B)) \
       + tf.reduce_mean(tf.abs(F_A - A)))

def residual_loss(X, G_X, F_GX, Y, F_Y, G_FY, delta_):
    real_noise = X - Y
    fake_noise = F_Y - G_X
    fake_noise_ = F_GX - G_FY
    return delta_ * (tf.reduce_mean(tf.abs(real_noise - fake_noise)) + tf.reduce_mean(tf.abs(real_noise - fake_noise_)))



#### cygle gan image pool
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp0 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            tmp1 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            tmp2 = copy.copy(self.images[idx])[2]
            self.images[idx][2] = image[2]
            tmp3 = copy.copy(self.images[idx])[3]
            self.images[idx][3] = image[3]
            return [tmp0, tmp1, tmp2, tmp3]
        else:
            return image