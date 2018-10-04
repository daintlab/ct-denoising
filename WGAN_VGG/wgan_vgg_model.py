# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:39:45 2018

@author: yeohyeongyu
"""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tcl
import numpy as np
import time
from glob import glob

import inout_util
import wgan_vgg_module as modules


class wganVgg(object):
    def __init__(self, sess, args):
        self.sess = sess    
        
        ####patients folder name
        self.train_patent_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') if ('zip' not in d) & (d not in args.test_patient_no)]     
        self.test_patent_no = args.test_patient_no    
        
        #### image placehold  (patch image, whole image)
        self.z_i= tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LDCT')
        self.x_i = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'NDCT')
        self.whole_z = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LDCT')
        self.whole_x = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_NDCT')

        #### set modules (generator, discriminator, vgg net)
        self.g_net = modules.generator
        self.d_net = modules.discriminator
        self.vgg = modules.Vgg19(vgg_path = args.pretrained_vgg) 
        

        #### generate & discriminate
        #generated images
        self.G_zi = self.g_net(self.z_i, reuse = False)
        self.G_whole_zi = self.g_net(self.whole_z)

        #discriminate
        self.D_xi = self.d_net(self.x_i, reuse = False)
        self.D_G_zi= self.d_net(self.G_zi)

        #### variable list
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]


        #### loss define
        #gradients penalty
        self.epsilon = tf.random_uniform([], 0.0, 1.0)
        self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
        self.D_x_hat = self.d_net(self.x_hat)
        self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]
        self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat), axis=1))
        self.gradient_penalty =  tf.square(self.grad_x_hat_l2 - 1.0)

        #perceptual loss
        self.G_zi_3c = tf.concat([self.G_zi]*3, axis=3)
        self.xi_3c = tf.concat([self.x_i]*3, axis=3)
        [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]
        self.vgg_perc_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((self.vgg.extract_feature(self.G_zi_3c) -  self.vgg.extract_feature(self.xi_3c))))) / (w*h*d))

        #discriminator loss(WGAN LOSS)
        d_loss = tf.reduce_mean(self.D_G_zi) - tf.reduce_mean(self.D_xi) 
        grad_penal =  args.lambda_ *tf.reduce_mean(self.gradient_penalty )
        self.D_loss = d_loss +grad_penal
        #generator loss
        self.G_loss = args.lambda_1 * self.vgg_perc_loss - tf.reduce_mean(self.D_G_zi)


        #### summary
        #loss summary
        self.summary_vgg_perc_loss = tf.summary.scalar("1_PerceptualLoss_VGG", self.vgg_perc_loss)
        self.summary_d_loss_all = tf.summary.scalar("2_DiscriminatorLoss_WGAN", self.D_loss)
        self.summary_d_loss_1 = tf.summary.scalar("3_D_loss_disc", d_loss)
        self.summary_d_loss_2 = tf.summary.scalar("4_D_loss_gradient_penalty", grad_penal)
        self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
        self.summary_all_loss = tf.summary.merge([self.summary_vgg_perc_loss, self.summary_d_loss_all, self.summary_d_loss_1, self.summary_d_loss_2, self.summary_g_loss])
            
        #psnr summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", inout_util.tf_psnr(self.whole_z, self.whole_x, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", inout_util.tf_psnr(self.whole_x, self.G_whole_zi, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])
        
        #image summary
        self.whole_img_summary = tf.concat([self.whole_z, self.whole_x, self.G_whole_zi], axis = 2)        
        self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)

        #ROI summary
        if args.is_mayo:
            self.ROI_real_zi =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_A')
            self.ROI_real_xi =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_B')
            self.ROI_fake_zi = self.g_net(self.ROI_real_zi)

            self.ROI_real_img_summary = tf.concat([self.ROI_real_zi, self.ROI_real_xi, self.ROI_fake_zi], axis = 2)
            self.summary_ROI_image_1 = tf.summary.image('2_ROI_image_1', self.ROI_real_img_summary)
            self.summary_ROI_image_2 = tf.summary.image('3_ROI_image_2', self.ROI_real_img_summary)
        
        
        #### optimizer
        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss, var_list = self.d_vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss, var_list = self.g_vars)
                
                
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    
        
        
        ###load images
        print('data load...') 
        self.image_loader = inout_util.DCMDataLoader(args.dcm_path, args.zi_image, args.xi_image, \
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size,\
             is_unpair = False, is_patch=True)
                                     
        self.test_image_loader = inout_util.DCMDataLoader(args.dcm_path, args.zi_image, args.xi_image,\
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = 1,\
             is_unpair = False, is_patch=False)
        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patent_no)
            
        self.test_image_loader(self.test_patent_no)
        print('data load complete !!!, {}'.format(time.time() - t1))
                
            
        print('--------------------------------------------\n# of parameters : {} '.\
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        
        
    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        self.start_step = 0
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
        print('Start point : iter : {}'.format(self.start_step))

        start_time = time.time()
        for t in range(self.start_step, args.num_iter):
            for _ in range(0, args.d_iters):
                #get random patches
                p_bz, p_bx = self.image_loader.batch_generator() #patch batches

                #discriminator update
                self.sess.run(self.d_adam, \
                              feed_dict={self.z_i : p_bz.reshape([-1] + self.z_i.get_shape().as_list()[1:]), \
                                         self.x_i : p_bx.reshape([-1] + self.x_i.get_shape().as_list()[1:])})

            #get random patches
            p_bz, p_bx = self.image_loader.batch_generator()     
            #generator update & loss summary
            _, summary_str= self.sess.run([self.g_adam, self.summary_all_loss], \
                                          feed_dict={self.z_i : p_bz.reshape([-1] + self.z_i.get_shape().as_list()[1:]), \
                                                     self.x_i : p_bx.reshape([-1] + self.x_i.get_shape().as_list()[1:])})
            self.writer.add_summary(summary_str, t)

            #print point
            if (t+1) % args.print_freq == 0:
                #print loss & time 
                d_loss, g_loss, g_zi_img = self.sess.run([self.D_loss, self.G_loss, self.G_zi], 
                                          feed_dict={self.z_i : p_bz.reshape([-1] + self.z_i.get_shape().as_list()[1:]), \
                                                     self.x_i : p_bx.reshape([-1] + self.x_i.get_shape().as_list()[1:])})
                
                print('Iter {} Time {} d_loss {} g_loss {}'.format(t, time.time() - start_time, d_loss, g_loss))

                #summary whole image'
                sltd_idx = np.random.choice(range(len(self.test_image_loader.zi_images)))
                test_zi, test_xi = self.test_image_loader.zi_images[sltd_idx], self.test_image_loader.xi_images[sltd_idx]

                whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})

                summary_str1, summary_str2= self.sess.run([self.summary_image, self.summary_psnr], \
                                         feed_dict={self.whole_z : test_zi.reshape(self.whole_z.get_shape().as_list()), \
                                                    self.whole_x : test_xi.reshape(self.whole_x.get_shape().as_list()), \
                                                    self.G_whole_zi : whole_G_zi.reshape(self.G_whole_zi.get_shape().as_list()),
                                                    })
                self.writer.add_summary(summary_str1, t)
                self.writer.add_summary(summary_str2, t)

                """
                    summary ROI IMAGE
                """
                if args.is_mayo:
                    ROI_sample = [['067', 676, 203, [161, 289], [61, 189]],
                                ['291', 103, 196, [111, 239], [111, 239]]]

                    ROI_LDCT_dir = ['/data1/Mayo-CT-3mm/test_input/L{}_3mm_test_input_{}.npy'.format(s[0], s[2]) for s in ROI_sample]
                    ROI_NDCT_dir = ['/data1/Mayo-CT-3mm/test_target/L{}_3mm_test_target_{}.npy'.format(s[0], s[2]) for s in ROI_sample]

                    RIO_LDCT_1, RIO_NDCT_1 = inout_util.load_test_image(ROI_LDCT_dir[0], ROI_NDCT_dir[0], image_max = args.img_vmax, image_min = args.img_vmin)
                    RIO_LDCT_2, RIO_NDCT_2 = inout_util.load_test_image(ROI_LDCT_dir[1], ROI_NDCT_dir[1], image_max = args.img_vmax, image_min = args.img_vmin)


                    ROI_LDCT_arr = [inout_util.ROI_img(RIO_LDCT_1,row = ROI_sample[0][3], col = ROI_sample[0][4]), \
                                    inout_util.ROI_img(RIO_LDCT_2,row = ROI_sample[1][3], col = ROI_sample[1][4])]

                    ROI_NDCT_arr = [inout_util.ROI_img(RIO_NDCT_1,row = ROI_sample[0][3], col = ROI_sample[0][4]), \
                                    inout_util.ROI_img(RIO_NDCT_2,row = ROI_sample[1][3], col = ROI_sample[1][4])]


                    ROI_fake_B_1 = self.sess.run(
                        self.ROI_fake_zi,  feed_dict={
                        self.ROI_real_zi :  ROI_LDCT_arr[0].reshape([1] + self.ROI_real_zi.get_shape().as_list()[1:])})

                    ROI_fake_B_2 = self.sess.run(
                        self.ROI_fake_zi,  feed_dict={
                        self.ROI_real_zi :  ROI_LDCT_arr[1].reshape([1] + self.ROI_real_zi.get_shape().as_list()[1:])})


                    ROI_fake_B_1, ROI_fake_B_2 = np.array(ROI_fake_B_1).astype(np.float32), np.array(ROI_fake_B_2).astype(np.float32)

                    roi_summary_str1 = self.sess.run(
                        self.summary_ROI_image_1,
                        feed_dict={self.ROI_real_zi : ROI_LDCT_arr[0].reshape([1] + self.ROI_real_zi.get_shape().as_list()[1:]), 
                                   self.ROI_real_xi : ROI_NDCT_arr[0].reshape([1] + self.ROI_real_xi.get_shape().as_list()[1:]),
                                   self.ROI_fake_zi: ROI_fake_B_1.reshape([1] + self.ROI_fake_zi.get_shape().as_list()[1:])})

                    roi_summary_str2  = self.sess.run(
                        self.summary_ROI_image_2,
                        feed_dict={self.ROI_real_zi : ROI_LDCT_arr[1].reshape([1] + self.ROI_real_zi.get_shape().as_list()[1:]), 
                                   self.ROI_real_xi : ROI_NDCT_arr[1].reshape([1] + self.ROI_real_xi.get_shape().as_list()[1:]),
                                   self.ROI_fake_zi: ROI_fake_B_2.reshape([1] + self.ROI_fake_zi.get_shape().as_list()[1:])})

                    self.writer.add_summary(roi_summary_str1, t)
                    self.writer.add_summary(roi_summary_str2, t)



            if (t+1) % args.save_freq == 0:
                self.save(t, args.checkpoint_dir)

    

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
        
        for idx in range(len(self.test_image_loader.zi_images)):
            test_zi, test_xi = self.test_image_loader.zi_images[idx], self.test_image_loader.xi_images[idx]
            
            whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})
            save_file_nm_f = 'from_' + self.test_image_loader.zi_image_pathes[idx].split('/')[-1]
            save_file_nm_t = 'to_' + self.test_image_loader.xi_image_pathes[idx].split('/')[-1]
            save_file_nm_g = 'Gen_from_' + self.test_image_loader.zi_image_pathes[idx].split('/')[-1]
            
            np.save(os.path.join(npy_save_dir, save_file_nm_f), test_zi)
            np.save(os.path.join(npy_save_dir, save_file_nm_t), test_xi)
            np.save(os.path.join(npy_save_dir, save_file_nm_g), whole_G_zi)
            
                
    def save(self, step, checkpoint_dir = 'checkpoint'):
        model_name = "wgan_vgg.model"
        checkpoint_dir = os.path.join('.', checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir = 'checkpoint'):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            
            self.start_step = int(ckpt_name.split('-')[-1])
            
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

