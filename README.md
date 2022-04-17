# ETC-RDSN
This repository is for A Novel Encryption-Then-Lossy-Compression Scheme of Color Images Using Customized Residual Dense Spatial Network(ETC-RDSN) introduced in the following paper.

Chuntao Wang, Tianjian Zhang, Hao Chen, Qiong Huang, Jiangqun Ni, Xinpeng Zhang"A Novel Encryption-Then-Lossy-Compression Scheme of Color Images Using Customized Residual Dense Spatial Network", IEEE Transactions on Multimedia(TMM), 2022.


If you use any part of our code, or ETC-RDSN is useful for your research, please consider citing::
```
@article{wang2022a,
  title={A Novel Encryption-Then-Lossy-Compression Scheme of Color Images Using Customized Residual Dense Spatial Network},
  author={Wang, Chuntao and Zhang, Tianjian and Chen, Hao and Huang, Qiong and Ni Jianqun and Zhang Xinpeng},
  journal={IEEE Transactions on Multimedia},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={IEEE}
}
```


## Abstract
Nowadays it has still remained as a big challenge to efficiently compress color images in the encrypted domain. In this paper we present a novel deep-learning-based approach to encryption-then-lossy-compression (ETC) of color images by incorporating the domain knowledge of the encrypted image reconstruction process. In specific, a simple yet effective uniform down-sampling is utilized for lossy compression of stream-ciphered images, and the task of image reconstruction from an encrypted down-sampled image is then formulated as a problem of constrained super-resolution (SR) reconstruction. A customized residual dense spatial network (RDSN) is proposed to solve the formulated constrained SR task by taking advantage of spatial attention mechanism (SAM), global skip connection(GSC), and uniform down-sampling constraint (UDC) that is specific to an ETC system. Extensive experimental results show that the proposed ETC scheme achieves significant performance improvement compared with other state-of-the-art ETC methods, indicating the feasibility and effectiveness of the proposed deep-learning based ETC scheme.

## Requirements
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* cv2 >= 3.xx



## Getting Started
### Installation
- Clone this repo:
```bash
git clone xxx
cd xxx
```

- Download weights from [Google Drive](https://drive.google.com/drive/folders/1XJueynAz4COLPbctbdvolpS21Jm5THde?usp=sharing).

### Train/Test
Download the DIV2K and benchmark dataset. More details about training and testing can be found in [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).
```bash
cd src
sh demo.sh
```

## Acknowledgments
Code largely benefits from [EDSR-Pytorch](https://github.com/sanghyun-son/EDSR-PyTorch). We thank the authors for sharing their codes. 