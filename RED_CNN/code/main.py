# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""


import argparse
import os
import tensorflow as tf
os.chdir(os.getcwd())
from inout_util import *
from red_cnn_model import  redCNN
os.chdir(os.getcwd() + '/..')
print('pwd : {}'.format(os.getcwd()))

parser = argparse.ArgumentParser(description='')
#set load directory
parser.add_argument('--dcm_path', dest='dcm_path', default= '/data1/AAPM-Mayo-CT-Challenge', help='dicom file directory')
parser.add_argument('--LDCT_path', dest='LDCT_path', default= 'quarter_3mm', help='LDCT image folder name')
parser.add_argument('--NDCT_path', dest='NDCT_path', default= 'full_3mm', help='NDCT image folder name')
parser.add_argument('--test_patient_no', dest='test_patient_no',type=ParseList, default= 'L067,L291')

#set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',  default='checkpoint', help='check point dir')
parser.add_argument('--test_npy_save_dir', dest='test_npy_save_dir',  default='./test/output_npy', help='test numpy file save dir')

#image info
parser.add_argument('--patch_size', dest='patch_size', type=int,  default=55, help='image patch size, h=w')
parser.add_argument('--whole_size', dest='whole_size', type=int,  default=512, help='image whole size, h=w')
parser.add_argument('--img_channel', dest='img_channel', type=int,  default=1, help='image channel, 1')
parser.add_argument('--img_vmax', dest='img_vmax', type=int,  default=3072, help='image max value, 3072')
parser.add_argument('--img_vmin', dest='img_vmin', type=int,  default=-1024, help='image max value -1024')

#train, test
parser.add_argument('--model', dest='model', default='red_cnn', help='red_cnn, wgan_vgg, cyclegan')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

#train detail
parser.add_argument('--num_iter', dest = 'num_iter', type = float, default = 200000, help = 'iterations')
parser.add_argument('--alpha', dest='alpha', type=float,  default=1e-4, help='learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int,  default=128, help='batch size')


#others
parser.add_argument('--mayo_roi', dest='mayo_roi', type=ParseBoolean, default=True, help='summary ROI sample1,2')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=2378, help='save a model every save_freq (iteration)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=500, help='print_freq (iterations)')
parser.add_argument('--continue_train', dest='continue_train', type=ParseBoolean, default=True, help='load the latest model: true, false')
parser.add_argument('--gpu_no', dest='gpu_no', type=int,  default=0, help='gpu no')

# -------------------------------------
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = redCNN(sess, args)
model.train(args) if args.phase == 'train' else model.test(args)