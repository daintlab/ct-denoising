from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

import module as md
import utils as ut


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess        
        #file directory list
        self.trainA_dirlist = sorted(glob(args.trainA_path + '/*.*'))
        self.trainB_dirlist = sorted(glob(args.trainB_path + '/*.*'))
        self.testA_dirlist = sorted(glob(args.testA_path + '/*.*'))
        self.testB_dirlist = sorted(glob(args.testB_path + '/*.*'))

        #module
        self.discriminator = md.discriminator
        self.generator = md.generator_resnet

        #network options
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.image_size,
                                      args.ngf, args.ndf, args.output_img_c,
                                      args.phase == 'train'))

        """
        build model
        """
        #real image placehold 
        self.real_A= tf.placeholder(tf.float32, [None, args.image_size, args.image_size, args.input_img_c], name='real_A')
        self.real_B = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, args.output_img_c], name='real_B')


        #### Generator & Discriminator
        #Generator
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        #Discriminator
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")

        #### Loss
        #generator loss
        self.cycle_loss = md.cycle_loss(self.real_A, self.fake_A_, self.real_B, self.fake_B_, args.L1_lambda)
        self.g_loss_a2b = md.least_square(self.DB_fake, tf.ones_like(self.DB_fake)) 
        self.g_loss_b2a = md.least_square(self.DA_fake, tf.ones_like(self.DA_fake)) 
        
        if args.resid_loss:
            self.residual_loss = md.residual_loss(self.real_A, self.fake_B, self.fake_A_, self.real_B, self.fake_A,  self.fake_B_, args.L1_lambda)
            self.g_loss = self.g_loss_a2b + self.g_loss_a2b + self.cycle_loss + self.residual_loss
        else:
            self.g_loss = self.g_loss_a2b + self.g_loss_a2b + self.cycle_loss 

        #dicriminator loss
        self.db_loss_real = md.least_square(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = md.least_square(self.DB_fake, tf.zeros_like(self.DB_fake))
        self.da_loss_real = md.least_square(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = md.least_square(self.DA_fake, tf.zeros_like(self.DA_fake))
        self.db_loss = (self.db_loss_real + self.db_loss_fake)
        self.da_loss = (self.da_loss_real + self.da_loss_fake)
        self.d_loss = (self.da_loss + self.db_loss) / 2
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        for var in t_vars: print(var.name)


        """
        Summary
        """
        #### loss summary
        #generator
        self.g_loss_sum = tf.summary.scalar("1_g_loss", self.g_loss, family = 'Generator_loss')
        self.cycle_loss_sum = tf.summary.scalar("2_cycle_loss", self.cycle_loss, family = 'Generator_loss')
        self.g_loss_a2b_sum = tf.summary.scalar("4_g_loss_a2b", self.g_loss_a2b, family = 'Generator_loss')
        self.g_loss_b2a_sum = tf.summary.scalar("5_g_loss_b2a", self.g_loss_b2a, family = 'Generator_loss')

        if args.resid_loss:
            self.residual_loss_sum = tf.summary.scalar("3_residual_loss", self.residual_loss, family = 'Generator_loss')
            self.g_sum = tf.summary.merge([self.g_loss_sum, self.cycle_loss_sum, self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.residual_loss_sum])
        else:
            self.g_sum = tf.summary.merge([self.g_loss_sum, self.cycle_loss_sum, self.g_loss_a2b_sum, self.g_loss_b2a_sum])            

        #discriminator
        self.d_loss_sum = tf.summary.scalar("1_d_loss", self.d_loss, family = 'Discriminator_loss')
        self.db_loss_real_sum = tf.summary.scalar("2_db_loss_real", self.db_loss_real, family = 'Discriminator_loss')
        self.db_loss_fake_sum = tf.summary.scalar("3_db_loss_fake", self.db_loss_fake, family = 'Discriminator_loss')
        self.d_sum = tf.summary.merge([self.d_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum])

        #### image summary
        self.real_img_summary = tf.concat([self.real_A, self.real_B, self.fake_B], axis = 2)
        self.summary_image_1 = tf.summary.image('1_train_whole_image', self.real_img_summary)
        self.summary_image_2 = tf.summary.image('2_test_whole_image', self.real_img_summary)

        #### image summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.real_A, self.real_B, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.real_B, self.fake_B, 2), family = 'PSNR')  #-1 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

        
        print('--------------------------------------------\n# of parameters : {} '.\
             format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        


        #for RIO summary
        if args.mayo_roi:
            #place hold
            self.ROI_real_A =  tf.placeholder(tf.float32, [None, 128, 128, args.input_img_c], name='ROI_A')
            self.ROI_real_B =  tf.placeholder(tf.float32, [None, 128, 128, args.output_img_c], name='ROI_B')
        
            #fakeB for ROI
            self.ROI_fake_B = self.generator(self.ROI_real_A, self.options, True, name="generatorA2B")

            #image summary
            self.ROI_real_img_summary = tf.concat([self.ROI_real_A, self.ROI_real_B, self.ROI_fake_B], axis = 2)
            self.summary_ROI_image_1 = tf.summary.image('3_ROI_image_1', self.ROI_real_img_summary)
            self.summary_ROI_image_2 = tf.summary.image('4_ROI_image_2', self.ROI_real_img_summary)
            #psnr summary
            self.summary_ROI_psnr_ldct_1 = tf.summary.scalar("3_ROI_psnr_LDCT_1", ut.tf_psnr(self.ROI_real_A, self.ROI_real_B, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_result_1 = tf.summary.scalar("4_ROI_psnr_output_1", ut.tf_psnr(self.ROI_real_B, self.ROI_fake_B, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_ldct_2 = tf.summary.scalar("5_ROI_psnr_LDCT_2", ut.tf_psnr(self.ROI_real_A, self.ROI_real_B, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_result_2 = tf.summary.scalar("6_ROI_psnr_output_2", ut.tf_psnr(self.ROI_real_B, self.ROI_fake_B, 2), family = 'PSNR')  #-1 ~ 1
            self.summary_ROI_psnr_1 = tf.summary.merge([self.summary_ROI_psnr_ldct_1, self.summary_ROI_psnr_result_1])
            self.summary_ROI_psnr_2 = tf.summary.merge([self.summary_ROI_psnr_ldct_2, self.summary_ROI_psnr_result_2])
       

        self.saver = tf.train.Saver(max_to_keep=None)
        self.pool = ut.ImagePool(args.max_size)


    def train(self, args):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        #optimizer
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())#tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        #pretrained model load
        start_time = time.time()
        self.start_step = 0 #load SUCESS -> self.start_step 파일명에 의해 초기화... // failed -> 0
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print('Start point : iter : {}'.format(self.start_step))

        #data load
        load_train_set = ut.DataLoader(self.trainA_dirlist , self.trainB_dirlist, \
            image_max = args.img_vmax, image_min = args.img_vmin, image_size = args.image_size, depth = args.input_img_c)
        A,B= load_train_set(shuffle=args.unpair)
        real_image_A, real_image_B =load_train_set.generate_image_batch(A,B, args.batch_size, shuffle = True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = self.sess)

        #iteration -> epoch
        self.start_epoch =  int(self.start_step / load_train_set.num_files_per_epoch)
        for epoch in range(self.start_epoch, args.end_epoch):
           
            batch_idxs = load_train_set.num_files_per_epoch // args.batch_size

            lr = args.lr / (10**int(epoch/ args.decay_epoch))
            
            for _ in range(0, batch_idxs):
                real_A, real_B = self.sess.run([real_image_A, real_image_B])

                # Update G network
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_A : real_A, self.real_B : real_B, self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)

                #image pool
                [real_A, real_B, fake_A, fake_B] = self.pool([real_A, real_B, fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_A : real_A, 
                               self.real_B : real_B,
                               self.fake_A : fake_A,
                               self.fake_B : fake_B,
                               self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)


                if self.start_step % args.print_freq == 0:
                    currt_step = self.start_step % load_train_set.num_files_per_epoch if epoch != 0 else self.start_step
                    print(("Epoch: {} {}/{} time: {} lr {}: ".format(epoch, currt_step, batch_idxs, time.time() - start_time, lr)))
                    #summary trainig sample image
                    summary_str1 = self.sess.run(
                        self.summary_image_1,
                        feed_dict={self.real_A : real_A.reshape([1] + self.real_A.get_shape().as_list()[1:]), 
                                   self.real_B : real_B.reshape([1] + self.real_B.get_shape().as_list()[1:]),
                                   self.fake_A : fake_A.reshape([1] + self.fake_A.get_shape().as_list()[1:]),
                                   self.fake_B : fake_B.reshape([1] + self.fake_B.get_shape().as_list()[1:])})
                    self.writer.add_summary(summary_str1, self.start_step)
                    
                    #summary test sample image
                    self.sample_model(args, epoch, self.start_step)

                if self.start_step % args.save_freq == 0:
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    self.save(args.checkpoint_dir, self.start_step)
                
                self.start_step += 1

        coord.request_stop()
        coord.join(threads)   

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

    #summary test sample image during training
    def sample_model(self, args, epoch, idx):
        sltd_idx = np.random.choice(range(len(self.testA_dirlist)))
        img_A, img_B = self.testA_dirlist[sltd_idx], self.testB_dirlist[sltd_idx]

        sample_A_image, sample_B_image = ut.load_test_image(img_A, img_B, image_max = args.img_vmax, image_min = args.img_vmin)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_B: sample_B_image.reshape([1] + self.real_B.get_shape().as_list()[1:]), 
            self.real_A: sample_A_image.reshape([1] + self.real_B.get_shape().as_list()[1:])})

        fake_A = np.array(fake_A).astype(np.float32)
        fake_B = np.array(fake_B).astype(np.float32)

        summary_str1, summary_str2 = self.sess.run(
            [self.summary_image_2, self.summary_psnr],
            feed_dict={self.real_A : sample_A_image.reshape([1] + self.real_A.get_shape().as_list()[1:]), 
                       self.real_B : sample_B_image.reshape([1] + self.real_B.get_shape().as_list()[1:]),
                       self.fake_A: fake_A.reshape([1] + self.fake_A.get_shape().as_list()[1:]),
                       self.fake_B: fake_B.reshape([1] + self.fake_B.get_shape().as_list()[1:])})
      
        self.writer.add_summary(summary_str1, idx)
        self.writer.add_summary(summary_str2, idx)  

        if args.mayo_roi:
            ROI_sample = [['067', 676, 203, [161, 289], [61, 189]],
                        ['291', 103, 196, [111, 239], [111, 239]]]
            
            ROI_LDCT_dir = ['/data/private/Mayo-CT-3mm/test_input/L{}_3mm_test_input_{}.npy'.format(s[0], s[2]) for s in ROI_sample]
            ROI_NDCT_dir = ['/data/private/Mayo-CT-3mm/test_target/L{}_3mm_test_target_{}.npy'.format(s[0], s[2]) for s in ROI_sample]

            RIO_LDCT_1, RIO_NDCT_1 = ut.load_test_image(ROI_LDCT_dir[0], ROI_NDCT_dir[0], image_max = args.img_vmax, image_min = args.img_vmin)
            RIO_LDCT_2, RIO_NDCT_2 = ut.load_test_image(ROI_LDCT_dir[1], ROI_NDCT_dir[1], image_max = args.img_vmax, image_min = args.img_vmin)

            
            ROI_LDCT_arr = [ut.ROI_img(RIO_LDCT_1, row = ROI_sample[0][3], col = ROI_sample[0][4]), \
                            ut.ROI_img(RIO_LDCT_2, row = ROI_sample[1][3], col = ROI_sample[1][4])]

            ROI_NDCT_arr = [ut.ROI_img(RIO_NDCT_1, row = ROI_sample[0][3], col = ROI_sample[0][4]), \
                            ut.ROI_img(RIO_NDCT_2, row = ROI_sample[1][3], col = ROI_sample[1][4])]
            
                
            ROI_fake_B_1 = self.sess.run(
                self.ROI_fake_B,  feed_dict={
                self.ROI_real_A :  ROI_LDCT_arr[0].reshape([1] + self.ROI_real_A.get_shape().as_list()[1:])})
            
            ROI_fake_B_2 = self.sess.run(
                self.ROI_fake_B,  feed_dict={
                self.ROI_real_A :  ROI_LDCT_arr[1].reshape([1] + self.ROI_real_A.get_shape().as_list()[1:])})
            
            
            roi_summary_str1, roi_summary_str2 = self.sess.run(
                [self.summary_ROI_image_1, self.summary_ROI_psnr_1],
                feed_dict={self.ROI_real_A : ROI_LDCT_arr[0].reshape([1] + self.ROI_real_A.get_shape().as_list()[1:]), 
                           self.ROI_real_B : ROI_NDCT_arr[0].reshape([1] + self.ROI_real_B.get_shape().as_list()[1:]),
                           self.ROI_fake_B: ROI_fake_B_1.reshape([1] + self.ROI_fake_B.get_shape().as_list()[1:])})
            
            roi_summary_str3, roi_summary_str4 = self.sess.run(
                [self.summary_ROI_image_2, self.summary_ROI_psnr_2],
                feed_dict={self.ROI_real_A : ROI_LDCT_arr[1].reshape([1] + self.ROI_real_A.get_shape().as_list()[1:]), 
                           self.ROI_real_B : ROI_NDCT_arr[1].reshape([1] + self.ROI_real_B.get_shape().as_list()[1:]),
                           self.ROI_fake_B: ROI_fake_B_2.reshape([1] + self.ROI_fake_B.get_shape().as_list()[1:])})
            self.writer.add_summary(roi_summary_str1, idx)
            self.writer.add_summary(roi_summary_str2, idx)
            self.writer.add_summary(roi_summary_str3, idx)
            self.writer.add_summary(roi_summary_str4, idx)
        
            
            
    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if not os.path.exists(os.path.join(args.test_dir, 'test_npy', 'AtoB')):
            os.makedirs(os.path.join(args.test_dir, 'test_npy', 'AtoB'))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for sample_A_file, sample_B_file in zip(self.testA_dirlist, self.testB_dirlist):
            sample_A_image, sample_B_image = ut.load_test_image(sample_A_file, sample_B_file, image_max = args.img_vmax, image_min = args.img_vmin)

            npy_A2B_dir =  os.path.join(args.test_dir, 'test_npy', '{}/{}'.format('AtoB', 'Gen_from_'+os.path.basename(sample_A_file)))

            mk_fake_B = self.sess.run([self.fake_B],
            feed_dict={self.real_A: sample_A_image.reshape([1] + self.real_A.get_shape().as_list()[1:])})


            mk_fake_B = np.array(mk_fake_B).astype(np.float32)

            np.save(npy_A2B_dir, mk_fake_B)

