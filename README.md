# DIP Term Project: Video Stabilization

- Group_21: “Deep Online Video Stabilization With Multi-Grid Warping Transformation Learning”

- This is a [PyTorch](http://pytorch.org/) implementation of Video Stabilization. 

    And we use Pixel-wise Warping Maps to improve the method of this paper.

    If you have any questions, please refer to github: https://github.com/mindazhao/PWStableNet


## Prerequisites

- Linux
- Python 3.6
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- pytorch 0.4.0+
- numpy
- cv2
- ...


## Testing

Put the test video at the folder named original.

Open terminal -> cd to pwstablenet -> python ./test.py  


### Download a pre-trained network
- We provide a pre-trained model.
- Torch models:
       https://drive.google.com/file/d/1PHeEVWUm5_D572ZEnE5_aWXjLO6ZMMNV/view?fbclid=IwAR1WdephRV_Jqry-z9_KDocm-XKXlKw5BE6ovTq2yKZYCsRsbztAP8qh6bw
- You can test your own unstable videos by changing the parameter "train"    with False and adjust the path yourself in function "process()".

## Training 
- Datasets :  DeepStab dataset (7.9GB) http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip  

- The code will download the [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at :             https://download.pytorch.org/models/vgg16-397923af.pth automatically.

- To train it using the train script simply specify the parameters listed in `./lib/cfg.py` as a flag or manually change them.
- The default parameters are set for the use of two NVIDIA 1080Ti graphic cards with 24G memory.
        CUDA_VISIBLE_DEVICES=0,1 python main_new.py


- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Before training, you should ensure the location of preprocessed dataset, which will be supplied soon.

## Reference 
- M. Wang, G.-Y. Yang, J.-K. Lin, S.-H. Zhang, A. Shamir, S.-P. Lu, and S.-M. Hu, “Deep online video stabilization with multi-grid warp- ing transformation learning,” IEEE Transactions on Image Processing, vol. 28, 2019
- M. Wang, Q. Ling, Senior Member, IEEE, “PWStableNet: Learning Pixel-Wise Warping
Maps for Video Stabilization,” IEEE Transactions on Image Processing, vol. 29, 2020
