# CYCLEGAN_BASE_MODEL-tensorflow
>	* CYCLE_GAN : https://github.com/hyeongyuy/ct-denoising/tree/master/cyclegan
>	* CYCLE_IDENTITY_GAN :https://github.com/hyeongyuy/ct-denoising/tree/master/CYCLE_IDENTITY_GAN
>	* reference code:  
>     * cyclegan : https://github.com/xhujoy/CycleGAN-tensorflow

## Training detail  
> * opt : Adam(learning rate = 0.0002, beta1 = 0.5, beta2 = 0.999)
> * learning rate decay : first 100 eppoch 0.0002 and linearly decreased it to zero over the next epochs.


## Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * LDCT_path : LDCT image folder name
> * NDCT_path : NDCT image folder name
> * test_patient_no : test patient id list(p_id1,p_id2...) (train patient id : (patient id list - test patient id list)
> * result : save result dir(check point, test, log, summary params)
> * checkpoint_dir : save directory - trained model
> * log_dir : save directory - tensoroard model
> * test_npy_save_dir : save directory - test numpy file
* Image info
> * patch_size : patch size 
> * whole_size : whole size
> * img_channel : image channel
> * img_vmax : max value
> * img_vmin : min value
* Train/Test
> * phase : train | test
* Training detail
> * img_pool : use image pool (default = True)
> * strct : cyc : resnet base cyclegan(CYCLEGAN), ident : identity loss paper (default = cyc)
> * augument : augumentation (default = False)
> * norm : normalization range, n-11 : -1 ~ 1, tanh(generator act-func), n01 : 0 ~ 1, sigmoid (default = n-11)
> * is_unpair : unpaired image (default = True)
> * max_size : image pool size (default = 50)
> * end_epoch : end epoch (default = 160)
> * lr : learning rate (default=0.0002)
> * batch_size : batch size (default=10)
> * L1_lambda : weight of cyclic loss (default=10)
> * L1_gamma : weight of identity loss (default=5)
> * beta1 : Adam optimizer parameter (default=0.5)
> * beta2 : Adam optimizer parameter (default=0.999)
> * ngf : # of generator filters in first conv layer
> * nglf : # of generator filters in last conv layer
> * ndf : # of discriminator filters in first conv layer
* loss params
> * cycle_loss : + cyclic loss (default = True)
> * ident_loss : + identity loss (default = False)
> * resid_loss : + residuel loss (default = True)
> * L1_lambda  : weight of cyclic loss
> * L1_gamma  : weight of identity loss
> * L1_delta  : weight of residual loss
* others
> * save_freq : save a model every step (iterations)
> * print_freq : print frequency (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)

## Run(example)
* train
> nohup python main.py --result=/model_set1 --end_epoch=10 --save_freq=200 --print_freq=50 --gpu_no=0 --resid_loss=True --ident_loss=False --L1_delta=10> set1 &

* test
> python main.py --result=/model_set1 --resid_loss=True --ident_loss=False --phase=test
