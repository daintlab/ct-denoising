# RED_CNN
## Network architecture  
![Network architecture](https://github.com/hyeongyuy/ct-denoising/blob/master/RED_CNN/img/architecture.JPG)  
* 10 layers (5 conv  + 5 deconv)
* shortcut
* remove pooling operation
* filter size : 96 * 9 + 1* 1(last layer)
* kernel size : 5 * 5
* stride : 1 (no padding)

## Training detail
* patch size : 55 * 55
* augumentation(patch? whole? // patch...)
>   * rotation(45 degrees)
>   * flipping (vertical & horizontal)
>   * scaling (0.5, 2)
* learning rate : 10e-4  (slowly decreased down(?))
* initializer : random Gaussian distribution (0, 0.01)
* loss function : MSE 
* optimizer : Adam 

## Main file(main.py) Parameters
* Training detail
> * num_iter : iterations (default = 200000)
> * alpha : learning rate (default=1e-4)
> * batch_size : batch size (default=128)

## Run
* train
> python main.py
* test
> python main.py --phase=test