import argparse
import os
import tensorflow as tf
from collections import namedtuple
import sys
os.chdir(os.getcwd())
sys.path.append('.')
from model import cyclegan
from module import *
from utils import *
os.chdir(os.getcwd() + '/..')

parser = argparse.ArgumentParser(description='')
# -------------------------------------
#directory (input)
parser.add_argument('--trainA_path', dest='trainA_path', default='/data/private/Mayo-CT-3mm/train_input',
                   help='train imageA path (LDCT)')
parser.add_argument('--trainB_path', dest='trainB_path', default='/data/private/Mayo-CT-3mm/train_target',
                   help='train imageB path (NDCT)')
parser.add_argument('--testA_path', dest='testA_path', default='/data/private/Mayo-CT-3mm/test_input',
                   help='test imageA path (LDCT)')
parser.add_argument('--testB_path', dest='testB_path', default='/data/private/Mayo-CT-3mm/test_target',
                   help='test imageB path (NDCT)')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test samples are saved here')


#train
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--end_epoch', dest='end_epoch', type=int, default=100, help='end epoch')
parser.add_argument('--decay_epoch', dest='decay_epoch', type=int, default=50, help='epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')


#image info
parser.add_argument('--image_size', dest='image_size', type=int, default=512, help='image  size')
parser.add_argument('--input_img_c', dest='input_img_c', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_img_c', dest='output_img_c', type=int, default=1, help='# of output image channels')
parser.add_argument('--img_vmax', dest='img_vmax', type=int, default=3072, help='max value in image')
parser.add_argument('--img_vmin', dest='img_vmin', type=int, default=-1024,  help='max value in image')


#others
parser.add_argument('--unpair', dest='unpair', type=ParseBoolean, default=False, help='unpaired image : True')
parser.add_argument('--resid_loss', dest='resid_loss', type=ParseBoolean, default=True, help='+ residuel loss : True')
parser.add_argument('--mayo_roi', dest='mayo_roi', type=ParseBoolean, default=True, help='summary ROI sample1,2')
parser.add_argument('--continue_train', dest='continue_train', type=ParseBoolean, default=True, help='load the latest model: true, false')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=2000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=200, help='print_freq iterations')

# -------------------------------------
args = parser.parse_args()
print(args)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = cyclegan(sess, args)
model.train(args) if args.phase == 'train' else model.test(args)

