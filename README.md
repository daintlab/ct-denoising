# ct-denoising
## Denoising Model
* RED_CNN
>	* paper :https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf
* WGAN_VGG
>	* paper : https://arxiv.org/pdf/1708.00961.pdf
>	* original code:  
>     * vgg : https://github.com/machrisaa/tensorflow-vgg  
>     * WGAN : https://github.com/jiamings/wgan
* CYCLEGAN
>	* paper : https://arxiv.org/abs/1703.10593
>	* original code: https://github.com/xhujoy/CycleGAN-tensorflow
## I/O (DICOM file -> .npy)
* Input data Directory  
  * DICOM file extension = [<b>'.IMA'</b>, '.dcm']
> $ os.path.join(dcm_path, patent_no, [LDCT_path|NDCT_path], '*.' + extension)
## [Common] Main file(main.py) Parameters
* Directory
> * dcm_path : dicom file directory
> * LDCT_path : LDCT image folder name
> * NDCT_path : NDCT image folder name
> * test_patient_no : default : L067,L291
> * checkpoint_dir : save directory - trained model
> * test_npy_save_dir : save directory - test numpy file
> * pretrained_vgg : pretrained vggnet directory(only WGAN_VGG)
* Image info
> * patch_size : image patch size (WGAN_VGG, RED_CNN)
> * whole_size : image whole size
> * img_channel : image channel
> * img_vmax : image
> * img_vmin : image
* Train/Test
> * model : red_cnn, wgan_vgg, cyclegan (for image preprocessing)
> * phase : train | test
* others
> * is_mayo : summary ROI sample1,2
> * save_freq : save a model every save_freq (iterations)
> * print_freq : print_freq (iterations)
> * continue_train : load the latest model: true, false
> * gpu_no : visible devices(gpu no)
