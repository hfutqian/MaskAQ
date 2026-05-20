### Requirements

* Python 3.9.18
* PyTorch 2.0.1
* Refer to the requirements.txt for other requirements


### Set the paths of datasets

Set the "dataPath" in "imagenet_deit_s_16_224.hocon" as the path root of your ImageNet dataset. For example:

        dataPath = "./dataset/ImageNet/"


### Training

To quantize the pre-trained DeiT-S on ImageNet to 3 bits:

    bash train.sh imagenet_deit_s_16_224.hocon 1234 0.001 3 3 ./gen_images_raw multi_step [50,100] lsq


