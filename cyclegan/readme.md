
# 파일
- main :  실행 코드
- model : Train, test관련된 코드
- module : generator, discriminator, loss등 모델 네트워크 관련 코드
- utils : dataloader, image pool, psnr 계산관련 함수

# 코드 실행
### directory:
main.py에  argparse 수정 (trainA_path, trainB_path, testA_path, testB_path)

## train
* paired cycle gan
> python main.py --unpair=False
* unpared cycle gan
> python main.py --unpair=True
* residual loss cycle gan
> python main.py --unpair=False --resid_loss=True

## test
> python main.py --phase=test

* 코드 원본
> https://github.com/xhujoy/CycleGAN-tensorflow

