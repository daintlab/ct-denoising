# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:45:16 2018

@author: yeohyeongyu
"""
import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from random import shuffle
import dicom

class DCMDataLoader(object):
    def __init__(self, dcm_path, zi_image_path, xi_image_path, \
                 image_size = 512, patch_size = 64,  depth = 1, \
                 image_max = 3072, image_min = -1024, batch_size = 128, \
                 is_unpair = False, is_patch = True):
        
        #dicom file dir
        self.dcm_path = dcm_path
        self.zi_image_path = zi_image_path
        self.xi_image_path = xi_image_path
        
        #image params
        self.image_min = image_min
        self.image_max = image_max 
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.depth = depth
        
        #training params
        self.batch_size = batch_size
        self.is_unpair = is_unpair
        self.is_patch = is_patch
        
    #dicom file -> numpy array
    def __call__(self, patent_no_list):
        self.zi_image_pathes = []
        self.xi_image_pathes = []
        for patent_no in patent_no_list:
            p_zi_pathes, p_xi_pathes =\
            glob(os.path.join(self.dcm_path, patent_no, self.zi_image_path, '*.IMA')), \
            glob(os.path.join(self.dcm_path, patent_no, self.xi_image_path, '*.IMA'))
            
            self.zi_image_pathes.extend(p_zi_pathes)
            self.xi_image_pathes.extend(p_xi_pathes)
        
        #load images
        org_zi_images = self.get_pixels_hu(self.load_scan(self.zi_image_pathes))
        org_xi_images = self.get_pixels_hu(self.load_scan(self.xi_image_pathes))
        
        #normalization  (0 ~ 1)
        self.zi_images = (org_zi_images - self.image_min )/(self.image_max - self.image_min)
        self.xi_images = (org_xi_images - self.image_min )/(self.image_max - self.image_min)
        

    def load_scan(self, path):
        slices = [dicom.read_file(s) for s in path]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness =\
            np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    #WGAN VGG
    def get_random_patch(self):
        img_h = img_w = self.image_size
        h = w =  self.patch_size
        size = [int(img_h / h), int(img_w / w)]
        
        self.zi_patches, self.xi_patches = [], []
        for zi, xi in zip(self.zi_batches, self.xi_batches):
            sltd_idx = np.random.choice(range(size[0] * size[1]))
            i = sltd_idx % size[0]
            j = sltd_idx // size[1]
            self.zi_patches.append(zi[j*h:j*h+h, i*w:i*w+w])
            self.xi_patches.append(xi[j*h:j*h+h, i*w:i*w+w])
        
    def batch_generator(self):
        if self.is_unpair:
            self.zi_index = np.random.choice(range(len(self.zi_images)), self.batch_size)
            self.xi_index = np.random.choice(range(len(self.xi_images)), self.batch_size)
        else:
            self.zi_index = self.xi_index = np.random.choice(range(len(self.zi_images)), self.batch_size) 
      
        self.zi_batches, self.xi_batches =  self.zi_images[self.zi_index], self.xi_images[self.xi_index]
        
        if self.is_patch:
            self.get_random_patch()
            return  np.array(self.zi_patches), np.array(self.xi_patches)
        else:
            return  self.zi_reshaped_img, self.xi_reshaped_img
    
    
def merge(images, whole_size = 512):
    h, w = images.shape[1], images.shape[2]
    size =  [int(whole_size / h), int(whole_size / w)]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


   

#ROI crop
def ROI_img(whole_image, row= [200, 350], col = [75, 225]):
   patch_ = whole_image[row[0]:row[1], col[0] : col[1]]
   return np.array(patch_)

#psnr
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
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


#save mk img
def save_image(LDCT, NDCT, output_, save_dir = '.',  max_ = 1, min_= 0): 
    f, axes  = plt.subplots(2, 3, figsize=(30, 20))
    
    axes[0,0].imshow(LDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[0,1].imshow(NDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[0,2].imshow(output_,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    
    axes[1,0].imshow(NDCT.astype(np.float32) - LDCT.astype(np.float32),  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[1,1].imshow(NDCT - output_,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[1,2].imshow(output_ - LDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    
    axes[0,0].title.set_text('LDCT image')
    axes[0,1].title.set_text('NDCT image')
    axes[0,2].title.set_text('output image')
    
    axes[1,0].title.set_text('NDCT - LDCT  image')
    axes[1,1].title.set_text('NDCT - outupt image')
    axes[1,2].title.set_text('output - LDCT  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()   


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
        
# argparser string -> boolean type
def ParseList(l):
    return l.split(',')
