"""
[X] : 논문이랑 다른 부분

* network structure
    1. generator detail
     - 8 conv layers (with relu)
     - kernel size : 3*3 
     - filters 
        - first_7 : 32
        - last : 1
    
    2. perceptual loss
     - pre-trained VGG net
     - feed : ground truth (generated image)
     - loss: L2 (ground truth)
     - gradient : only generator
    
    3. discriminator
     -str :  
         - 6 conv layer(with leaky relu)
         -> fully connected : 1024 (wit leaky relu)
         -> fully connected : 1 (cross entropy)
     - filters:
         - first_2 : 64
         - mid_2 : 128
         - last_2  : 256
         - kernel size : 3*3
         
-------------------------------------------------
    * data(in article... )
        - [X]random extract patch (pairs):  
                train : 100,096 (from 4,000)
                test : 5,056 ( from 2,000)
          -> 따로 저장하지 않고, 모델에 들어가기 전에 patch 형태로 변환
        - size : 64*64 
        - [X]remove mostly air images 
          -> 처리 X(원본 이미지를 0~1사이 값으로 변환해서 입력으로...)
    
-------------------------------------------------
* training detail
 - mini batch size L : 128
 - opt : Adam(alpha = 1e-5, beta1 = 0.5, beta2 = 0.9)
 - discriminator iter : 4
 - [X] epoch : 100   <- iteration으로 변경
 - lambda(WGAN weight penalty) : 10
 - lambda1(VGG weight) : 0.1, lambda2 : 0.1
 
-----------------------------------------------------------------------------------------
"""

### *** ROI summary 부분 실행 X(파일이름을 원래 파일 이름 그대로 사용...)


## 실행 예시
train)<br>
$python main.py --dcm_path=/data1/AAPM-Mayo-CT-Challenge --zi_image=quarter_3mm --xi_image=full_3mm --test_patient_no=L067,L291
test)<br>
$python main.py --dcm_path=/data1/AAPM-Mayo-CT-Challenge --zi_image=quarter_3mm --xi_image=full_3mm --test_patient_no=L067,L291 --phase=test


(train에 사용되는 환자번호는 test 환자번호 제외한 나머지(--test_patient_no)


## source
vgg : https://github.com/machrisaa/tensorflow-vgg
WGAN : https://github.com/jiamings/wgan