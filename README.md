
## Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers [ICML 2026]
This repository is the official code for the paper "Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers" by Biao Qian, Yang Wang, Yong Wu and Jungong Han.



### Requirements

* Python 3.9.18
* PyTorch 2.0.1
* Refer to the requirements.txt for other requirements


### Set the paths of datasets

Set the "dataPath" in "imagenet_deit_s_16_224.hocon" as the path root of your ImageNet dataset. For example:

        dataPath = "./dataset/ImageNet/"


### Training

For example, to quantize the pre-trained DeiT-S on ImageNet to 3 bits, please run:

    bash train.sh imagenet_deit_s_16_224.hocon 1234 0.001 3 3 ./gen_images_raw multi_step [50,100] lsq



## Citation
If you find the project codes useful for your research, please consider citing
```
@article{qian2026selective,
  title={Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers},
  author={Qian, Biao and Wang, Yang and Wu, Yong and Han, Jungong},
  journal={arXiv preprint arXiv:2606.04373},
  year={2026}
}
```


