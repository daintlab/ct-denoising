# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import cyclegan_module as md
import inout_util as ut
from random import shuffle

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess        
        
        ####patients folder name
        self.train_patient_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') \
         if ('zip' not in d) & (d.split('/')[-1] not in args.test_patient_no)]     
        self.test_patient_no = args.test_patient_no    

        
        #save directory
        self.p_info = '_'.join(self.test_patient_no)
        self.checkpoint_dir = os.path.join(args.result, args.checkpoint_dir, self.p_info)
        self.log_dir = os.path.join(args.result, args.log_dir,  self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(\
                     self.checkpoint_dir, self.log_dir))

        #module
        self.discriminator = md.discriminator
        self.generator = md.generator_resnet

        """
        load images
        """
        print('data load... dicom -> numpy') 
        self.image_loader = ut.DCMDataLoader(\
              args.dcm_path, args.LDCT_path, args.NDCT_path, \
             image_size = args.whole_size, patch_size = args.patch_size, \
             depth = args.img_channel, image_max = args.img_vmax, image_min = args.img_vmin,\
             is_unpair = args.is_unpair, augument = args.augument, norm = args.norm)
                                     
        self.test_image_loader = ut.DCMDataLoader(\
             args.dcm_path, args.LDCT_path, args.NDCT_path,\
             image_size = args.whole_size, patch_size = args.patch_size, \
              depth = args.img_channel, image_max = args.img_vmax, image_min = args.img_vmin,\
             is_unpair = args.is_unpair, augument = args.augument, norm = args.norm)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patient_no)
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(\
                 time.time() - t1, len(self.image_loader.LDCT_image_name), \
                len(self.test_image_loader.LDCT_image_name)))
        else:
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}, N_test : {}'.format(\
              time.time() - t1, len(self.test_image_loader.LDCT_image_name)))
        

        """
        build model
        """
        #### image placehold 
        self.real_X =  tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name = 'LDCT')
        self.real_Y =  tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name = 'NDCT')
        self.sample_GX =  tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name = 'G_LDCT')
        self.sample_FY =  tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name = 'F_NDCT')
        self.X = tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name='X')
        self.Y = tf.placeholder(tf.float32, \
         [None, args.patch_size, args.patch_size, args.img_channel], name='Y')


        #### Generator & Discriminator
        #Generator
        self.G_X = self.generator(args, self.real_X, False, name="generatorX2Y")
        self.F_GX = self.generator(args, self.G_X, False, name="generatorY2X")
        self.F_Y = self.generator(args, self.real_Y, True, name="generatorY2X")
        self.G_FY = self.generator(args, self.F_Y, True, name="generatorX2Y")
        
        #Discriminator
        self.D_GX = self.discriminator(args, self.G_X, reuse=False, \
                                   name="discriminatorY") #for generator loss
        self.D_FY = self.discriminator(args, self.F_Y, reuse=False, \
                                   name="discriminatorX") #for generator loss
        self.D_sample_GX = self.discriminator(args, self.sample_GX, \
                              reuse=True, name="discriminatorY") #for discriminator loss
        self.D_sample_FY = self.discriminator(args, self.sample_FY, \
                          reuse=True, name="discriminatorX") #for discriminator loss
        self.D_Y = self.discriminator(args, self.real_Y, reuse=True, \
                                  name="discriminatorY")
        self.D_X = self.discriminator(args, self.real_X, reuse=True, \
                                  name="discriminatorX")
        
        
        #### Loss
        #generator loss
        self.cycle_loss = md.cycle_loss(\
            self.real_X, self.F_GX, self.real_Y, self.G_FY, args.L1_lambda_1)
        self.G_loss_X2Y = md.least_square(self.D_GX, tf.ones_like(self.D_GX)) 
        self.G_loss_Y2X = md.least_square(self.D_FY, tf.ones_like(self.D_FY)) 
        
        if args.resid_loss:
            self.residual_loss = md.residual_loss(\
                      self.real_X, self.G_X, self.F_GX, self.real_Y, \
                      self.F_Y,  self.G_FY, args.L1_lambda_2)
            self.G_loss = self.G_loss_X2Y + self.G_loss_Y2X \
            + self.cycle_loss + self.residual_loss
        else:
            self.G_loss = self.G_loss_X2Y + self.G_loss_Y2X + self.cycle_loss 

        #dicriminator loss
        self.D_loss_real_Y = md.least_square(self.D_Y, tf.ones_like(self.D_Y))
        self.D_loss_GX = md.least_square(self.D_sample_GX, tf.zeros_like(self.D_sample_GX))
        self.D_loss_real_X = md.least_square(self.D_X, tf.ones_like(self.D_X))
        self.D_loss_FY = md.least_square(self.D_sample_FY, tf.zeros_like(self.D_sample_FY))
        self.D_loss_Y = (self.D_loss_real_Y + self.D_loss_GX)
        self.D_loss_X = (self.D_loss_real_X + self.D_loss_FY)
        self.D_loss = (self.D_loss_X + self.D_loss_Y) / 2
    
        #### variable list
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        
        #### optimizer
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)
        
        """
        Summary
        """
        #### loss summary
        #generator
        self.G_loss_sum = tf.summary.scalar("1_G_loss", self.G_loss, family = 'Generator_loss')
        self.cycle_loss_sum = tf.summary.scalar("2_cycle_loss", self.cycle_loss, \
                                family = 'Generator_loss')
        self.G_loss_X2Y_sum = tf.summary.scalar("3_G_loss_X2Y", self.G_loss_X2Y, \
                                family = 'Generator_loss')
        self.G_loss_Y2X_sum = tf.summary.scalar("4_G_loss_Y2X", self.G_loss_Y2X, \
                                family = 'Generator_loss')

        if args.resid_loss:
            self.residual_loss_sum = tf.summary.scalar("5_residual_loss", self.residual_loss, family = 'Generator_loss')
            self.g_sum = tf.summary.merge(\
              [self.G_loss_sum, self.cycle_loss_sum, self.G_loss_X2Y_sum, \
               self.G_loss_Y2X_sum, self.residual_loss_sum])
        else:
            self.g_sum = tf.summary.merge(\
             [self.G_loss_sum, self.cycle_loss_sum, \
              self.G_loss_X2Y_sum, self.G_loss_Y2X_sum])            

        #discriminator
        self.D_loss_sum = tf.summary.scalar("1_D_loss", self.D_loss, \
                          family = 'Discriminator_loss')
        self.D_loss_Y_sum = tf.summary.scalar("2_D_loss_Y", self.D_loss_Y, \
                          family = 'Discriminator_loss')
        self.D_loss_GX_sum = tf.summary.scalar("3_D_loss_GX", self.D_loss_GX, \
                          family = 'Discriminator_loss')
        self.d_sum = tf.summary.merge([self.D_loss_sum, self.D_loss_Y_sum, self.D_loss_GX_sum])

        #### image summary
        self.test_G_X = self.generator(args, self.X, True, name="generatorX2Y")
        self.train_img_summary = tf.concat([self.real_X, self.real_Y, self.G_X], axis = 2)
        self.summary_image_1 = tf.summary.image('1_train_whole_image', self.train_img_summary)
        self.test_img_summary = tf.concat([self.X, self.Y, self.test_G_X], axis = 2)
        self.summary_image_2 = tf.summary.image('2_test_whole_image', self.test_img_summary)

        #### psnr summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", \
         ut.tf_psnr(self.X, self.Y, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", \
         ut.tf_psnr(self.Y, self.test_G_X, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, \
         self.summary_psnr_result])

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)
        #image pool
        self.pool = md.ImagePool(args.max_size)


        print('--------------------------------------------\n# of parameters : {} '.\
             format(np.sum([np.prod(v.get_shape().as_list()) \
                        for v in tf.trainable_variables()])))
        

    def train(self, args):
        init_op = tf.group(tf.global_variables_initializer(), \
               tf.local_variables_initializer())
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        #pretrained model load
        self.start_step = 0 #load SUCESS -> initialize by file name // failed -> 0
        if args.continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")


        #iteration -> epoch
        self.start_epoch =  \
          int((self.start_step + 1) / len(self.image_loader.LDCT_image_name))
        epoch_size = len(self.image_loader.LDCT_image_name)
        print('Start point : iter : {}, epoch : {}'.format(\
           self.start_step, self.start_epoch))    
        start_time = time.time()
        lr = args.lr
        for epoch in range(self.start_epoch, args.end_epoch):
           
            #decay learning rate
            if epoch > args.decay_epoch:
                lr = args.lr - (epoch - (args.decay_epoch)) \
                  * ((args.lr / (args.end_epoch - args.decay_epoch)))
       
            #shuffling whole image index
            if args.is_unpair:
                shuffle(self.image_loader.LDCT_index)
                shuffle(self.image_loader.NDCT_index)
            else:
                self.image_loader.NDCT_index = self.image_loader.LDCT_index
                shuffle(self.LDCT_index)                         
                
            for _ in range(0, epoch_size):
                #input images(patch, augumentation...)
                real_sample_X, real_sample_Y  =  self.image_loader.preproc_input(args)

                # Update G network
                X, Y, F_Y, G_X, _, summary_str = self.sess.run(
                    [self.real_X , self.real_Y, self.F_Y, self.G_X, self.g_optim, self.g_sum], \
                    feed_dict = {self.real_X : real_sample_X, \
                                 self.real_Y : real_sample_Y, \
                                 self.lr:lr})

                self.writer.add_summary(summary_str, self.start_step)

                #image pool
                [X, Y, F_Y, G_X] = self.pool([X, Y, F_Y, G_X])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],  feed_dict = {self.real_X:X , self.real_Y:Y, self.sample_GX:G_X, self.sample_FY:F_Y, self.lr:lr})

                self.writer.add_summary(summary_str, self.start_step)


                if (self.start_step+1) % args.print_freq == 0:
                    currt_step = self.start_step\
                            % len(self.image_loader.LDCT_image_name) \
                            if epoch != 0 else self.start_step
                    print(("Epoch: {} {}/{} time: {} lr {}: ".format(\
                         epoch, currt_step, epoch_size, time.time() - start_time, lr)))
                    
                    #summary trainig sample image
                    summary_str1 = self.sess.run(self.summary_image_1, \
                                 feed_dict = {self.real_X:X , self.real_Y:Y, self.G_X:G_X})
                    
                    self.writer.add_summary(summary_str1, self.start_step)
                    
                    #check sample image
                    self.check_sample(args, self.start_step)

                if (self.start_step+1) % args.save_freq == 0:
                    self.save(args, self.start_step)
                
                self.start_step += 1
        self.save(args, self.start_step)
        
    #summary test sample image during training
    def check_sample(self, args, idx):
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_image_name)))

        #get test iamge(patch..)
        sample_X_image, sample_Y_image  \
        = self.test_image_loader.get_randam_patches(\
            self.test_image_loader.LDCT_images[sltd_idx], \
            self.test_image_loader.NDCT_images[sltd_idx], args.patch_size)


        G_X = self.sess.run(
            self.test_G_X,
            feed_dict={self.Y: sample_Y_image.reshape(\
              [1] + self.Y.get_shape().as_list()[1:]), 
            self.X: sample_X_image.reshape(\
              [1] + self.Y.get_shape().as_list()[1:])})

        G_X = np.array(G_X).astype(np.float32)

        summary_str1, summary_str2 = self.sess.run(
            [self.summary_image_2, self.summary_psnr],
            feed_dict={self.X : sample_X_image.reshape(\
                           [1] + self.X.get_shape().as_list()[1:]), 
                       self.Y : sample_Y_image.reshape(\
                           [1] + self.Y.get_shape().as_list()[1:]),
                       self.test_G_X: G_X.reshape(\
                           [1] + self.test_G_X.get_shape().as_list()[1:])})
      
        self.writer.add_summary(summary_str1, idx)
        self.writer.add_summary(summary_str2, idx)  

    # save model    
    def save(self, args, step):
        self.checkpoint_dir = os.path.join('.', self.checkpoint_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, ".model"),
                        global_step=step)

    # load model    
    def load(self):
        print(" [*] Reading checkpoint...")
        self.checkpoint_dir = os.path.join('.', self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

            
    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        npy_save_dir = os.path.join(args.result, args.test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)
            
            
        ## test
        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_X = self.test_image_loader.LDCT_images[idx]
            
            mk_G_X = self.sess.run(self.test_G_X, feed_dict={self.X: test_X.reshape([1] + self.X.get_shape().as_list()[1:])})
            
            save_file_nm_g = 'Gen_from_' + self.test_image_loader.LDCT_image_name[idx]
            
            np.save(os.path.join(npy_save_dir, save_file_nm_g), mk_G_X)
