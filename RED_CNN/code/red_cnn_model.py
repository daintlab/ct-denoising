# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018

@author: yeohyeongyu
"""


import os
import tensorflow as tf
import numpy as np
import time
from glob import glob
import red_cnn_module as modules
import inout_util as ut

class redCNN(object):
    def __init__(self, sess, args):
        self.sess = sess    
        
        ####patients folder name
        self.train_patent_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') if ('zip' not in d) & (d not in args.test_patient_no)]     
        self.test_patent_no = args.test_patient_no    
        
        #### set modules
        self.red_cnn = modules.redcnn
        
        """
        load images
        """
        print('data load... dicom -> numpy') 
        self.image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path, \
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, model = args.model)
                                     
        self.test_image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path,\
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, model = args.model)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patent_no)
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1, len(self.image_loader.LDCT_image_name), len(self.test_image_loader.LDCT_image_name)))
            [self.X, self.Y] = self.image_loader.input_pipeline(self.sess, args.patch_size, args.num_iter)
        else:
            self.test_image_loader(self.test_patent_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1, len(self.test_image_loader.LDCT_image_name)))
            self.X = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LDCT')
            self.Y = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'NDCT')
        
        """
        build model
        """
        self.whole_X = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LDCT')
        self.whole_Y = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_NDCT')

        #### denoised images
        self.output_img = self.red_cnn(self.X, reuse = False)
        self.WHOLE_output_img  = self.red_cnn(self.whole_X)
        
        #### loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.output_img))

        #### trainable variable list
        self.t_vars = tf.trainable_variables()

        #### optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(args.alpha, self.global_step, args.num_iter, args.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss, var_list = self.t_vars, global_step=self.global_step)

        """
        summary
        """
        #loss summary
        self.summary_loss = tf.summary.scalar("loss", self.loss)
        #psnr summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.whole_Y, self.whole_X, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.whole_Y, self.WHOLE_output_img, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])
        
        #image summary
        self.check_img_summary = tf.concat([tf.expand_dims(self.X[0], axis=0), \
                                    tf.expand_dims(self.Y[0], axis=0), \
                                    tf.expand_dims(self.output_img[0], axis=0)], axis = 2)  
        self.summary_train_image = tf.summary.image('0_train_image', self.check_img_summary)                                    
        self.whole_img_summary = tf.concat([self.whole_X, self.whole_Y, self.WHOLE_output_img], axis = 2)        
        self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)

        #ROI summary
        if args.mayo_roi:
            self.ROI_X =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_X')
            self.ROI_Y =  tf.placeholder(tf.float32, [None, 128, 128, args.img_channel], name='ROI_Y')
            self.ROI_output = self.red_cnn(self.ROI_X)

            self.ROI_real_img_summary = tf.concat([self.ROI_X, self.ROI_Y, self.ROI_output], axis = 2)
            self.summary_ROI_image_1 = tf.summary.image('2_ROI_image_1', self.ROI_real_img_summary)
            self.summary_ROI_image_2 = tf.summary.image('3_ROI_image_2', self.ROI_real_img_summary)
        

                
        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    

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
            #summary loss
            _, summary_str= self.sess.run([self.optimizer, self.summary_loss])
            self.writer.add_summary(summary_str, t)

            #print point
            if (t+1) % args.print_freq == 0:
                #print loss & time 
                loss, output_img, summary_str0 = self.sess.run([self.loss, self.output_img, self.summary_train_image])
                print('Iter {} Time {} loss {}'.format(t, time.time() - start_time, loss))
                #training sample check
                self.writer.add_summary(summary_str0, t)
                
                
                #check sample image
                self.check_sample(args, t)

            if (t+1) % args.save_freq == 0:
                self.save(t, args.checkpoint_dir)

        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(self.image_loader.enqueue_threads)
        
        
    #summary test sample image during training
    def check_sample(self, args, t):
        #summary whole image'
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_images)))
        test_X, test_Y = self.test_image_loader.LDCT_images[sltd_idx], self.test_image_loader.NDCT_images[sltd_idx]

        WHOLE_output_img = self.sess.run(self.WHOLE_output_img, feed_dict={self.whole_X: test_X.reshape(self.whole_X.get_shape().as_list())})

        summary_str1, summary_str2= self.sess.run([self.summary_image, self.summary_psnr], \
                                 feed_dict={self.whole_X : test_X.reshape(self.whole_X.get_shape().as_list()), \
                                            self.whole_Y : test_Y.reshape(self.whole_Y.get_shape().as_list()), \
                                            self.WHOLE_output_img : WHOLE_output_img.reshape(self.WHOLE_output_img.get_shape().as_list()),
                                            })
        self.writer.add_summary(summary_str1, t)
        self.writer.add_summary(summary_str2, t)

        """
            summary ROI IMAGE
        """
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
            
            ROI_output_1 = self.sess.run(
                self.ROI_output,  feed_dict={
                self.ROI_X :  ROI_LDCT_arr[0].reshape([1] + self.ROI_X.get_shape().as_list()[1:])})

            ROI_output_2 = self.sess.run(
                self.ROI_output,  feed_dict={
                self.ROI_X :  ROI_LDCT_arr[1].reshape([1] + self.ROI_X.get_shape().as_list()[1:])})


            ROI_output_1, ROI_output_2 = np.array(ROI_output_1).astype(np.float32), np.array(ROI_output_2).astype(np.float32)

            roi_summary_str1 = self.sess.run(
                self.summary_ROI_image_1,
                feed_dict={self.ROI_X : ROI_LDCT_arr[0].reshape([1] + self.ROI_X.get_shape().as_list()[1:]), 
                           self.ROI_Y : ROI_NDCT_arr[0].reshape([1] + self.ROI_Y.get_shape().as_list()[1:]),
                           self.ROI_output: ROI_output_1.reshape([1] + self.ROI_output.get_shape().as_list()[1:])})

            roi_summary_str2  = self.sess.run(
                self.summary_ROI_image_2,
                feed_dict={self.ROI_X : ROI_LDCT_arr[1].reshape([1] + self.ROI_X.get_shape().as_list()[1:]), 
                           self.ROI_Y : ROI_NDCT_arr[1].reshape([1] + self.ROI_Y.get_shape().as_list()[1:]),
                           self.ROI_output: ROI_output_2.reshape([1] + self.ROI_output.get_shape().as_list()[1:])})

            self.writer.add_summary(roi_summary_str1, t)
            self.writer.add_summary(roi_summary_str2, t)


    
    def save(self, step, checkpoint_dir = 'checkpoint'):
        model_name = "red_cnn.model"
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
        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_X, test_Y = self.test_image_loader.LDCT_images[idx], self.test_image_loader.NDCT_images[idx]
            
            WHOLE_output_img = self.sess.run(self.WHOLE_output_img, feed_dict={self.whole_X: test_X.reshape(self.whole_X.get_shape().as_list())})
            
            save_file_nm_f = 'from_' +  self.test_image_loader.LDCT_image_name[idx]
            save_file_nm_t = 'to_' +  self.test_image_loader.NDCT_image_name[idx]
            save_file_nm_g = 'Gen_from_' +  self.test_image_loader.LDCT_image_name[idx]
            
            np.save(os.path.join(npy_save_dir, save_file_nm_f), test_X)
            np.save(os.path.join(npy_save_dir, save_file_nm_t), test_Y)
            np.save(os.path.join(npy_save_dir, save_file_nm_g), WHOLE_output_img)
            
                
