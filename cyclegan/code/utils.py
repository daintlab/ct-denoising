# -*- coding: utf-8 -*-
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import numpy as np
import copy
import tensorflow as tf


class DataLoader(object):
    def __init__(self, imageA_path, imageB_path, image_max = 3072, image_min = -1024, image_size = 512, depth = 1):
        #directory
        self.imageA_path = imageA_path
        self.imageB_path = imageB_path

        #capacity...
        self.min_fraction_of_examples_in_queue =  1
        
        #image params
        self.image_max = image_max
        self.image_min=  image_min
        
        self.height, self.width = image_size, image_size
        self.depth = depth
        self.image_byte_size = self.height * self.width * self.depth *2 + 64*2  #int16 -> 2byte -> *2 & header?? 
        
    #make file name queue
    def __call__(self, shuffle):
        if len(self.imageA_path) != len(self.imageB_path) : print('# of image A != # of iamge B')

        self.num_files_per_epoch = min(len(self.imageA_path), len(self.imageB_path))
        self.min_queue_examples =  int(self.min_fraction_of_examples_in_queue * self.num_files_per_epoch )
        
        imgA_filename_queue = tf.train.string_input_producer(self.imageA_path, shuffle = shuffle)
        imgB_filename_queue = tf.train.string_input_producer(self.imageB_path, shuffle = shuffle)
        
        imgA_dequeue, imgB_dequeue = self.dequeue_image(imgA_filename_queue), self.dequeue_image(imgB_filename_queue )
        return imgA_dequeue, imgB_dequeue 

    def dequeue_image(self, file_queue):
        # define reader
        reader = tf.FixedLengthRecordReader(record_bytes=self.image_byte_size)
        key,value = reader.read(file_queue)
        
        #define decoder
        image_bytes = tf.decode_raw(value, tf.int16)
        image_bytes_ = tf.transpose(tf.reshape(tf.strided_slice(image_bytes, [64], [64 + self.image_byte_size]), [self.depth, self.height, self.width]),  [1, 2, 0])
        reshaped_image = tf.cast(image_bytes_, tf.float32)
        reshaped_image = ((reshaped_image - self.image_min )/(self.image_max - self.image_min) -0.5) * 2 # -1 ~ 1
        return reshaped_image

    def generate_image_batch(self, A_img_queue, B_img_queue, batch_size, shuffle):
        num_preprocess_threads = 16
        if shuffle:
            batch_imagesA, batch_imagesB = tf.train.shuffle_batch(
                [A_img_queue, B_img_queue],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size,
                min_after_dequeue=self.min_queue_examples)
        else:
            batch_imagesA, batch_imagesB = tf.train.batch(
                [A_img_queue, B_img_queue],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size)
            
        return  batch_imagesA, batch_imagesB  

# -----------------------------
def load_test_image(img_A_path, img_B_path, image_max = 3072, image_min = -1024):

    img_A = np.load(img_A_path).astype(np.float)
    img_B = np.load(img_B_path).astype(np.float)
    
    if image_max == None:
        image_max = np.max(img_A.reshape(-1))

    img_A = ((img_A - image_min )/(image_max - image_min) -0.5) * 2
    img_B = ((img_B - image_min )/(image_max - image_min) -0.5) * 2
       
    img_A = np.expand_dims(img_A, axis = 2)
    img_B = np.expand_dims(img_B, axis = 2)

    return img_A, img_B


#----------------------
# new added functions for cyclegan
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

#---------------------------------------------------
#psnr
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype = numerator.dtype))
  return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))


def psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#---------------------------------------------------
#ROI crop
def ROI_img(whole_image, row = [200, 350], col = [75, 225]):
   patch_ = whole_image[row[0]:row[1], col[0] : col[1]]
   return np.array(patch_)


def recon_img(img, max_ = 3072, min_=-1024):
    img = (img + 1) /2
    img = img * (max_ - min_) + min_
    return img.astype(int)

def trunc(array_):
    arr = array_.copy()
    arr[arr <= -160] = -160
    arr[arr >= 240] = 240
    return arr

#---------------------------------------------------
# argparser string -> boolean type
def ParseBoolean (b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError ('Cannot parse string into boolean.')