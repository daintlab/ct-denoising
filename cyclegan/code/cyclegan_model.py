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


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess        
        
        ####patients folder name
        self.train_patent_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') if ('zip' not in d) & (d not in args.test_patient_no)]     
        self.test_patent_no = args.test_patient_no    

        #module
        self.discriminator = md.discriminator
        self.generator = md.generator_resnet

        #network options
        OPTIONS = namedtuple('OPTIONS', 'image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.whole_size,
                                      args.ngf, args.ndf, args.img_channel,
                                      args.phase == 'train'))

        """
        build model
        """
        #real image placehold 
        self.X = tf.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='X')
        self.Y = tf.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='Y')


        #### Generator & Discriminator
        #Generator
        self.G_X = self.generator(self.X, self.options, False, name="generatorX2Y")
        self.F_GX = self.generator(self.G_X, self.options, False, name="generatorY2X")
        self.F_Y = self.generator(self.Y, self.options, True, name="generatorY2X")
        self.G_FY = self.generator(self.F_Y, self.options, True, name="generatorX2Y")
        #Discriminator
        self.D_GX = self.discriminator(self.G_X, self.options, reuse=False, name="discriminatorY")
        self.D_FY = self.discriminator(self.F_Y, self.options, reuse=False, name="discriminatorX")
        self.D_Y = self.discriminator(self.Y, self.options, reuse=True, name="discriminatorY")
        self.D_X = self.discriminator(self.X, self.options, reuse=True, name="discriminatorX")

        #### Loss
        #generator loss
        self.cycle_loss = md.cycle_loss(self.X, self.F_GX, self.Y, self.G_FY, args.L1_lambda)
        self.G_loss_X2Y = md.least_square(self.D_GX, tf.ones_like(self.D_GX)) 
        self.G_loss_Y2X = md.least_square(self.D_FY, tf.ones_like(self.D_FY)) 
        
        if args.resid_loss:
            self.residual_loss = md.residual_loss(self.X, self.G_X, self.F_GX, self.Y, self.F_Y,  self.G_FY, args.L1_lambda)
            self.G_loss = self.G_loss_X2Y + self.G_loss_X2Y + self.cycle_loss + self.residual_loss
        else:
            self.G_loss = self.G_loss_X2Y + self.G_loss_X2Y + self.cycle_loss 

        #dicriminator loss
        self.D_loss_real_Y = md.least_square(self.D_Y, tf.ones_like(self.D_Y))
        self.D_loss_GX = md.least_square(self.D_GX, tf.zeros_like(self.D_GX))
        self.D_loss_real_X = md.least_square(self.D_X, tf.ones_like(self.D_X))
        self.D_loss_FY = md.least_square(self.D_FY, tf.zeros_like(self.D_FY))
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
        self.cycle_loss_sum = tf.summary.scalar("2_cycle_loss", self.cycle_loss, family = 'Generator_loss')
        self.G_loss_X2Y_sum = tf.summary.scalar("3_G_loss_X2Y", self.G_loss_X2Y, family = 'Generator_loss')
        self.G_loss_Y2X_sum = tf.summary.scalar("4_G_loss_Y2X", self.G_loss_Y2X, family = 'Generator_loss')

        if args.resid_loss:
            self.residual_loss_sum = tf.summary.scalar("5_residual_loss", self.residual_loss, family = 'Generator_loss')
            self.g_sum = tf.summary.merge([self.G_loss_sum, self.cycle_loss_sum, self.G_loss_X2Y_sum, self.G_loss_Y2X_sum, self.residual_loss_sum])
        else:
            self.g_sum = tf.summary.merge([self.G_loss_sum, self.cycle_loss_sum, self.G_loss_X2Y_sum, self.G_loss_Y2X_sum])            

        #discriminator
        self.D_loss_sum = tf.summary.scalar("1_D_loss", self.D_loss, family = 'Discriminator_loss')
        self.D_loss_Y_sum = tf.summary.scalar("2_D_loss_Y", self.D_loss_Y, family = 'Discriminator_loss')
        self.D_loss_GX_sum = tf.summary.scalar("3_D_loss_GX", self.D_loss_GX, family = 'Discriminator_loss')
        self.d_sum = tf.summary.merge([self.D_loss_sum, self.D_loss_Y_sum, self.D_loss_GX_sum])

        #### image summary
        self.real_img_summary = tf.concat([self.X, self.Y, self.G_X], axis = 2)
        self.summary_image_1 = tf.summary.image('1_train_whole_image', self.real_img_summary)
        self.summary_image_2 = tf.summary.image('2_test_whole_image', self.real_img_summary)

        #### image summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.X, self.Y, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.Y, self.G_X, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

        
        #for RIO summary
        if args.mayo_roi:
            #place hold
            self.ROI_X =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_X')
            self.ROI_Y =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_Y')
        
            #fakeB for ROI
            self.ROI_GX = self.generator(self.ROI_X, self.options, True, name="generatorX2Y")

            #image summary
            self.ROI_real_img_summary = tf.concat([self.ROI_X, self.ROI_Y, self.ROI_GX], axis = 2)
            self.summary_ROI_image_1 = tf.summary.image('3_ROI_image_1', self.ROI_real_img_summary)
            self.summary_ROI_image_2 = tf.summary.image('4_ROI_image_2', self.ROI_real_img_summary)
            #psnr summary
            self.summary_ROI_psnr_ldct_1 = tf.summary.scalar("3_ROI_psnr_LDCT_1", ut.tf_psnr(self.ROI_X, self.ROI_Y, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_result_1 = tf.summary.scalar("4_ROI_psnr_output_1", ut.tf_psnr(self.ROI_Y, self.ROI_GX, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_ldct_2 = tf.summary.scalar("5_ROI_psnr_LDCT_2", ut.tf_psnr(self.ROI_X, self.ROI_Y, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_result_2 = tf.summary.scalar("6_ROI_psnr_output_2", ut.tf_psnr(self.ROI_Y, self.ROI_GX, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_1 = tf.summary.merge([self.summary_ROI_psnr_ldct_1, self.summary_ROI_psnr_result_1])
            self.summary_ROI_psnr_2 = tf.summary.merge([self.summary_ROI_psnr_ldct_2, self.summary_ROI_psnr_result_2])
       
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)
        #image pool
        self.pool = md.ImagePool(args.max_size)


        print('--------------------------------------------\n# of parameters : {} '.\
             format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        
        """
        load images
        """
        print('data load...') 
        self.image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path, \
             image_size = args.whole_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin,\
             is_unpair = args.unpair, model = args.model)
                                     
        self.test_image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path,\
             image_size = args.whole_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin,\
             is_unpair = args.unpair, model = args.model)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patent_no)
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1, len(self.image_loader.LDCT_image_name), len(self.test_image_loader.LDCT_image_name)))
        else:
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1, len(self.test_image_loader.LDCT_image_name)))
        
        
    def train(self, args):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        #pretrained model load
        self.start_step = 0 #load SUCESS -> self.start_step 파일명에 의해 초기화... // failed -> 0
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print('Start point : iter : {}'.format(self.start_step))

        #iteration -> epoch
        self.start_epoch =  int(self.start_step / len(self.image_loader.LDCT_image_name))

        start_time = time.time()
        for epoch in range(self.start_epoch, args.end_epoch):
           
            batch_idxs = len(self.image_loader.LDCT_image_name)

            lr = args.lr / (10**int(epoch/ args.decay_epoch))
            
            for _ in range(0, batch_idxs):
                
               #get images
                X, Y = self.image_loader.batch_generator() #patch batches
                X, Y = X.reshape([-1] + self.X.get_shape().as_list()[1:]), \
                                Y.reshape([-1] + self.Y.get_shape().as_list()[1:]), \
                                                
                # Update G network
                F_Y, G_X, _, summary_str = self.sess.run(
                    [self.F_Y, self.G_X, self.g_optim, self.g_sum],
                    feed_dict={self.X : X,
                               self.Y : Y, self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)

                #image pool
                [X, Y, F_Y, G_X] = self.pool([X, Y, F_Y, G_X])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.X : X, 
                               self.Y : Y,
                               self.F_Y : F_Y,
                               self.G_X : G_X,
                               self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)


                if (self.start_step+1) % args.print_freq == 0:
                    currt_step = self.start_step % len(self.image_loader.LDCT_image_name) if epoch != 0 else self.start_step
                    print(("Epoch: {} {}/{} time: {} lr {}: ".format(epoch, currt_step, batch_idxs, time.time() - start_time, lr)))
                    
                    #summary trainig sample image
                    summary_str1 = self.sess.run(
                        self.summary_image_1,
                        feed_dict={self.X : X.reshape([1] + self.X.get_shape().as_list()[1:]), 
                                   self.Y : Y.reshape([1] + self.Y.get_shape().as_list()[1:]),
                                   self.F_Y : F_Y.reshape([1] + self.F_Y.get_shape().as_list()[1:]),
                                   self.G_X : G_X.reshape([1] + self.G_X.get_shape().as_list()[1:])})
                    self.writer.add_summary(summary_str1, self.start_step)
                    
                    #check sample image
                    self.check_sample(args, self.start_step)

                if (self.start_step+1) % args.save_freq == 0:
                    self.save(args.checkpoint_dir, self.start_step)
                
                self.start_step += 1

    #summary test sample image during training
    def check_sample(self, args, idx):
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_image_name)))

        sample_X_image, sample_Y_image  = self.test_image_loader.LDCT_images[sltd_idx], self.test_image_loader.NDCT_images[sltd_idx]


        F_Y, G_X = self.sess.run(
            [self.F_Y, self.G_X],
            feed_dict={self.Y: sample_Y_image.reshape([1] + self.Y.get_shape().as_list()[1:]), 
            self.X: sample_X_image.reshape([1] + self.Y.get_shape().as_list()[1:])})

        F_Y = np.array(F_Y).astype(np.float32)
        G_X = np.array(G_X).astype(np.float32)

        summary_str1, summary_str2 = self.sess.run(
            [self.summary_image_2, self.summary_psnr],
            feed_dict={self.X : sample_X_image.reshape([1] + self.X.get_shape().as_list()[1:]), 
                       self.Y : sample_Y_image.reshape([1] + self.Y.get_shape().as_list()[1:]),
                       self.F_Y: F_Y.reshape([1] + self.F_Y.get_shape().as_list()[1:]),
                       self.G_X: G_X.reshape([1] + self.G_X.get_shape().as_list()[1:])})
      
        self.writer.add_summary(summary_str1, idx)
        self.writer.add_summary(summary_str2, idx)  

        if args.mayo_roi:
            ROI_sample = [['067', '0203', [161, 289], [61, 189]],
                        ['291', '0196', [111, 239], [111, 239]]]
            
            LDCT_ROI_idx = [self.test_image_loader.LDCT_image_name.index(\
                'L{}_{}_{}'.format(s[0], args.LDCT_path, s[1])) for s in ROI_sample]
            
            NDCT_ROI_idx = [self.test_image_loader.NDCT_image_name.index(\
                'L{}_{}_{}'.format(s[0], args.NDCT_path, s[1])) for s in ROI_sample]


            RIO_LDCT  = [self.test_image_loader.LDCT_images[idx] for idx in LDCT_ROI_idx]
            RIO_NDCT  = [self.test_image_loader.NDCT_images[idx] for idx in NDCT_ROI_idx]

            ROI_LDCT_arr = [ut.ROI_img(RIO_LDCT[0], row = ROI_sample[0][2], col = ROI_sample[0][3]), \
                            ut.ROI_img(RIO_LDCT[1], row = ROI_sample[1][2], col = ROI_sample[1][3])]

            ROI_NDCT_arr = [ut.ROI_img(RIO_NDCT[0], row = ROI_sample[0][2], col = ROI_sample[0][3]), \
                            ut.ROI_img(RIO_NDCT[1], row = ROI_sample[1][2], col = ROI_sample[1][3])]
            
                
            ROI_GX_1 = self.sess.run(
                self.ROI_GX,  feed_dict={
                self.ROI_X :  ROI_LDCT_arr[0].reshape([1] + self.ROI_X.get_shape().as_list()[1:])})
            
            ROI_GX_2 = self.sess.run(
                self.ROI_GX,  feed_dict={
                self.ROI_X :  ROI_LDCT_arr[1].reshape([1] + self.ROI_X.get_shape().as_list()[1:])})
            
            
            roi_summary_str1, roi_summary_str2 = self.sess.run(
                [self.summary_ROI_image_1, self.summary_ROI_psnr_1],
                feed_dict={self.ROI_X : ROI_LDCT_arr[0].reshape([1] + self.ROI_X.get_shape().as_list()[1:]), 
                           self.ROI_Y : ROI_NDCT_arr[0].reshape([1] + self.ROI_Y.get_shape().as_list()[1:]),
                           self.ROI_GX: ROI_GX_1.reshape([1] + self.ROI_GX.get_shape().as_list()[1:])})
            
            roi_summary_str3, roi_summary_str4 = self.sess.run(
                [self.summary_ROI_image_2, self.summary_ROI_psnr_2],
                feed_dict={self.ROI_X : ROI_LDCT_arr[1].reshape([1] + self.ROI_X.get_shape().as_list()[1:]), 
                           self.ROI_Y : ROI_NDCT_arr[1].reshape([1] + self.ROI_Y.get_shape().as_list()[1:]),
                           self.ROI_GX: ROI_GX_2.reshape([1] + self.ROI_GX.get_shape().as_list()[1:])})
            self.writer.add_summary(roi_summary_str1, idx)
            self.writer.add_summary(roi_summary_str2, idx)
            self.writer.add_summary(roi_summary_str3, idx)
            self.writer.add_summary(roi_summary_str4, idx)
        
                
    # save model    
    def save(self, checkpoint_dir,  step):
        model_name = "cyclegan.model"
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # load model    
    def load(self, checkpoint_dir = 'checkpoint'):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

            
    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        npy_save_dir = os.path.join('.', args.test_npy_save_dir)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)
            
            
        ## test
        start_time = time.time()
        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_X, test_Y  = self.test_image_loader.LDCT_images[idx], self.test_image_loader.NDCT_images[idx]
            
            mk_G_X = self.sess.run(self.G_X, feed_dict={self.X: test_X.reshape([1] + self.X.get_shape().as_list()[1:])})
            
            save_file_nm_g = 'Gen_from_' + self.test_image_loader.LDCT_image_name[idx]
            
            np.save(os.path.join(npy_save_dir, save_file_nm_g), mk_G_X)
