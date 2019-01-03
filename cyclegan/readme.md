# CYCLEGAN
## Archtecture & training detail  
![Network architecture](https://github.com/hyeongyuy/ct-denoising/blob/master/cyclegan/img/cyclegan_arch_detail.jpg)  
## Cycle-consistency  
![cycle loss](https://github.com/hyeongyuy/ct-denoising/blob/master/cyclegan/img/cycle_loss_concept.JPG)  
## Objective & loss  
![Objective](https://github.com/hyeongyuy/ct-denoising/blob/master/cyclegan/img/cyclegan_loss.jpg)  
## Main file(main.py) Parameters
* training detail
> * end_epoch : end epoch (default=200)
> * decay_epoch : epoch to decay lr (default=100)
> * lr : initial learning rate for adam (default=0.0002)
> * beta1 : momentum term of adam (default=0.5)
> * L1_lambda : weight on L1 term in objective (default=10)
> * max_size : max size of image pool (default=50)
> * ngf : N generator filter in first conv layer (default=32)
> * ndf : N discriminator filters in first conv layer (default=64)
* others
> * unpair : unpaired image[True|False]
> * resid_loss : + residuel loss: [True|False]
## Run
* train(unpair)
> python main.py --unpair=True (default=False)
* train(add residual loss)
> python main.py --resid_loss=True (default=False)
* test
> python main.py --phase=test