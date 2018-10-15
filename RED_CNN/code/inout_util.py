# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""

import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import dicom
from random import shuffle

class DCMDataLoader(object):
    def __init__(self, dcm_path, LDCT_image_path, NDCT_image_path, \
                 image_size = 512, patch_size = 64,  depth = 1, \
                 image_max = 3072, image_min = -1024, batch_size = 1, \
                 is_unpair = False, model='', extension = 'IMA'):
        
        #dicom file dir
        self.extension = extension
        self.dcm_path = dcm_path
        self.LDCT_image_path = LDCT_image_path
        self.NDCT_image_path = NDCT_image_path
        
        #image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth

        self.image_max = image_max
        self.image_min = image_min
        
        #training params
        self.batch_size = batch_size
        self.is_unpair = is_unpair
        self.model = model
        
        #CT slice name
        self.LDCT_image_name, self.NDCT_image_name = [], []

        #N call batch generator
        self.n_call_batgen =0
        
    #dicom file -> numpy array
    def __call__(self, patent_no_list):
        p_LDCT = []
        p_NDCT = []
        for patent_no in patent_no_list:
            P_LDCT_path, p_NDCT_path =\
            glob(os.path.join(self.dcm_path, patent_no, self.LDCT_image_path, '*.' + self.extension)), \
            glob(os.path.join(self.dcm_path, patent_no, self.NDCT_image_path, '*.' + self.extension))
            
            #load images
            org_LDCT_images, LDCT_slice_nm  = self.get_pixels_hu(self.load_scan(P_LDCT_path), '{}_{}'.format(patent_no, self.LDCT_image_path))
            org_NDCT_images, NDCT_slice_nm  = self.get_pixels_hu(self.load_scan(p_NDCT_path), '{}_{}'.format(patent_no, self.NDCT_image_path))
     
            #CT slice name
            self.LDCT_image_name.extend(LDCT_slice_nm)
            self.NDCT_image_name.extend(NDCT_slice_nm)

            #normalization  
            p_LDCT.append(normalize(org_LDCT_images, self.image_max , self.image_min, self.model))
            p_NDCT.append(normalize(org_NDCT_images, self.image_max , self.image_min, self.model))
            
        self.LDCT_images = np.concatenate(tuple(p_LDCT), axis=0)
        self.NDCT_images = np.concatenate(tuple(p_NDCT), axis=0)
             
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

    def get_pixels_hu(self, slices, pre_fix_nm = ''):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0

        digit = 4
        slice_nm = []
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)

            # sorted(idx), sorted(d_idx)  -> [1, 10, 2], [ 0001, 0002, 0010]
            s_idx = str(slice_number)
            d_idx = '0'*(digit - len(s_idx)) + s_idx
            slice_nm.append(pre_fix_nm + '_' + d_idx)
        return np.array(image, dtype=np.int16), slice_nm

    """
    get overlapped random patches(WGAN_VGG, RED_CNN)
    """
    def get_randam_patches(self, whole_size, patch_size, batch_size, model='wgan_vgg'):
        def augumentation(LDCT, NDCT):
            sltd_idx = np.random.choice(range(3))
            h,w = LDCT.shape
            # rotation
            if sltd_idx == 0 : 
                return [rotate(LDCT, 45, reshape = False), rotate(NDCT, 45, reshape = False)]
            #flipping
            elif sltd_idx == 1:
                flp = np.random.choice([True, False])
                if flp: #horizontal
                    return [LDCT[:, ::-1], NDCT[:, ::-1]]
                else: # vertical 
                    return [LDCT[::-1, :], NDCT[::-1, :]]
            #scaling            
            else: 
                sf =  np.random.choice([0.5, 2])
                return [LDCT * sf, NDCT * sf]

            
        whole_h =  whole_w = whole_size
        h = w = patch_size

        #patch image range
        hd = h//2  
        hu = int(whole_h - np.round(h/2))
        wd = w//2  
        wu = int(whole_w - np.round(w/2))

        LDCT_patches, NDCT_patches = [], [] 

        if model.lower() == 'red_cnn':
            for i in range(batch_size):
                #select random image(whole image)
                slice_idx = np.random.choice(range(len(self.LDCT_images)))
                LDCT, NDCT = self.LDCT_images[slice_idx], self.NDCT_images[slice_idx]

                #patch image center(coordinate on whole image)
                h_pc, w_pc  = np.random.choice(range(hd, hu+1)), np.random.choice(range(wd, wu+1))

                #augumentation
                [p_LDCT, p_NDCT] = augumentation(LDCT[h_pc - hd : int(h_pc + np.round(h/2)), w_pc - wd : int(w_pc + np.round(w/2))], \
                             NDCT[h_pc - hd : int(h_pc + np.round(h/2)), w_pc - wd : int(w_pc + np.round(w/2))])
                
                LDCT_patches.append(p_LDCT)
                NDCT_patches.append(p_NDCT)
                
        else:
            for i in range(batch_size):
                #select random image(whole image)
                slice_idx = np.random.choice(range(len(self.LDCT_images)))
                LDCT, NDCT = self.LDCT_images[slice_idx], self.NDCT_images[slice_idx]

                #patch image center(coordinate on whole image)
                h_pc, w_pc  = np.random.choice(range(hd, hu+1)), np.random.choice(range(wd, wu+1))
                LDCT_patches.append(LDCT[h_pc - hd : int(h_pc + np.round(h/2)), w_pc - wd : int(w_pc + np.round(h/2))])
                NDCT_patches.append(NDCT[h_pc - hd : int(h_pc + np.round(h/2)), w_pc - wd : int(w_pc + np.round(h/2))])
                
        return np.array(LDCT_patches), np.array(NDCT_patches)
        
        
    def normalize(self, img, max_ = 3072, min_=-1024, model ='cyclegan'):
        if model == 'cyclegan':  #-1 ~ 1
            img = 2 * ((img - min_) / (max_  -  min_)) -1
            return img
        else: # 0 ~ 1
            img = (img - min_) / (max_  -  min_)
            return img

    def batch_generator(self):
        if self.model.lower() == 'cyclegan':
            def new_shuffle():
                if self.is_unpair:
                    shuffle(self.LDCT_index)
                    shuffle(self.NDCT_index)
                else:
                    shuffle(self.LDCT_index)        

            if self.n_call_batgen == 0:
                if self.is_unpair:
                    self.LDCT_index, self.NDCT_index = list(range(len(self.LDCT_images))), list(range(len(self.NDCT_images)))
                else:
                    self.LDCT_index =  self.NDCT_index = list(range(len(self.LDCT_images)))
                new_shuffle()    
                self.LDCT_generator, self.NDCT_generator= iter(self.LDCT_images[self.LDCT_index]), iter(self.NDCT_images[self.NDCT_index])
                self.n_call_batgen +=1

            try:
                return next(self.LDCT_generator), next(self.NDCT_generator)
            except StopIteration:
                new_shuffle()    
                self.LDCT_generator, self.NDCT_generator= iter(self.LDCT_images[self.LDCT_index]), iter(self.NDCT_images[self.NDCT_index])
                return next(self.LDCT_generator), next(self.NDCT_generator)
        else:
            return self.get_randam_patches(self.image_size, self.patch_size, self.batch_size, model = self.model)


def normalize(img, max_ = 3072, min_=-1024, model ='cyclegan'):
    if model == 'cyclegan':  #-1 ~ 1
        img = 2 * ((img - min_) / (max_  -  min_)) -1
        return img
    else: # 0 ~ 1
        img = (img - min_) / (max_  -  min_)
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