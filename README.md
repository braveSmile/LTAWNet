> 引入轻量级Transformer的自适应窗口立体匹配算法  
> LTAW: Adaptive Window Stereo Matching Algorithm with Lightweight Transformer  
> [Wang Zhengjia,  Hu Feifei,  Zhang Chengjuan,  Lei Zhuo,  He Tao.]
> 2022

网络概览  
This is an overview of the network
![](media\pipeline.png)

KITTI数据集上的结果：  
Fine-tuned result on dataset of KITTI:
![](media\KITTI2015IMAGE.png)

仅在KITTI数据上训练时泛化到街道场景  
Generalizes to street scenes when trained on KITTI data only:
![](media\street_scenes.png)


## 介绍 Introduction
**动机 Motivation：**  
- LTAWNet(Lightweight Transformer Adaptive Window Net)是一种端到端的立体匹配算法，主要解决现有立体匹配方法存在匹配精度和运行效率难以平衡的问题。现有专注于高精度的立体匹配算法，比如基于3D卷积和基于相关性的立体匹配算法，会设置固定视差范围来减轻内存和计算需求，却是以损失精度为代价，而没有视差范围局限的Transformer特征描述匹配算法存在较高的延迟，即便其所提出的线性注意力版本的Transformer依然存在较大的计算消耗。为此提出了一种基于轻量级Transformer自适应窗口的立体匹配算法。  
LTAWNet(Lightweight Transformer Adaptive Window Net) is an end-to-end stereo matching algorithm, which mainly solves the problem that the matching accuracy and operation efficiency of the existing stereo matching methods are difficult to balance. The existing stereo matching algorithms focusing on high accuracy, such as 3D convolution based and correlation based stereo matching algorithms, set a fixed disparity range to reduce memory and computing requirements, but at the cost of loss of accuracy. The Transformer feature description matching algorithm without disparity range limitation has high delay. Even the proposed linear attention version of Transformer still has large computational consumption. This paper proposed a stereo matching algorithm based on lightweight Transformer adaptive window.  

**LTAW的优点 Benefits of LTAW：**  
- 视差范围随图像分辨率自然缩放，不再需要手动设置视差范围。  
The disparity range scales naturally with the image resolution and it is no longer necessary to manually set the disparity range.
使用自适应窗口匹配细化方法在保证匹配精度的同时有更高的执行效率。  
The adaptive window matching refinement method has higher execution efficiency while ensuring the matching accuracy.  
在立体匹配网络中引入了可分离自注意力层和坐标注意力层使网络进一步轻量化。
A separable self-attention layer and a coordinate attention layer are introduced into the stereo matching network to further lightweight the network.

## Dependencies
We recommend the following steps to set up your environment
- Create your python virtual environment by 
    ``` sh
    conda create --name ltaw python=3.8 # create a virtual environment called "ltaw" with python version 3.8
    ```
    (Python version >= 3.6)
- **Install Pytorch**: Please follow link [here](https://pytorch.org/get-started/locally/).

     (PyTorch version >= 1.5.1)
  
- **Other third-party packages**: You can use pip to install the dependencies by 
    ```sh
    pip install -r requirements.txt
    ``` 
- **(*Optional*) Install Nvidia apex**: We use apex for mixed precision training to accelerate training. To install, please follow instruction [here](https://github.com/NVIDIA/apex)
    - You can **remove** apex dependency if 
        - you have more powerful GPUs, or
        - you don't need to run the training script.
    - Note: We tried to use the native mixed precision training from official Pytorch implementation. However, it currently does *not* support *gradient checkpointing* for **LayerNorm**. We will post update if this is resolved.

## Pre-trained Models
You can download the pretrained model from the following links.
- Baidu download link:
  - Link: https://pan.baidu.com/s/1Ii_lmFwrjF1mtgSOMFTROA 
  - Password: `if98 `

## Folder Structure
#### Code Structure
```
stereo-transformer
    |_ configs (data configs )
    |_ src (network modules,  loss, dataloder)
    |_ utilities (training, evaluation, inference, logger etc.)
```
#### Data Structure
Please see [sample_data](sample_data) folder for details. We keep the original data folder structure from the official site. If you need to modify the existing structure, make sure to modify the dataloader.

- Note: We only provide one sample of each dataset to run the code. We do not own any copyright or credits of the data.

Scene Flow 
```
SCENE_FLOW
    |_ RGB_finalpass
        |_ TRAIN
            |_ A
                |_0000
    |_ disparity
        |_ TRAIN
            |_ A
                |_0000
    |_ occlusion
        |_ TRAIN
            |_ left
```
KITTI 2015
```
KITTI_2015
    |_ training
        |_ disp_occ_0 (disparity including occluded region)
        |_ image_2 (left image)
        |_ image_3 (right image)
``` 

## Usage
If you have a GPU and want to run locally:
- Download pretrained model using links in Pre-trained Models.
  - Note: The pretrained model is assumed to be in the LTAW-master folder.
- An example(ltaw_test.py) of how to run inference is given in file 

#### Terminal Example
- Download pretrained model using links in Pre-trained Models
- Run pretraining by
    ```
    sh scripts/pretrain.sh
    ```
    - Note: please set the `--dataset_directory` argument in the `.sh` file to where Scene Flow data is stored, i.e. replace `PATH_TO_SCENEFLOW`
- Run fine-tune on KITTI by
    ```
    sh scripts/kitti_finetune.sh
    ```
    - Note: please set the `--dataset_directory` argument in the `.sh` file to where KITTI data is stored, i.e. replace `PATH_TO_KITTI`
    - Note: the pretrained model is assumed to be in the `stereo-transformer` folder. 
- Run evaluation on the provided KITTI example by
    ```
    sh scripts/kitti_toy_eval.sh
    ```
    - Note: the pretrained model is assumed to be in the `stereo-transformer` folder. 

## Expected Result
The result of STTR may vary by a small fraction depending on the trial, but it should be approximately the same as the tables below.

Sceneflow

|            	|    **3px Error** 	|       **EPE**     | **Occ IOU**     |
|:----------:	|:---------------:	|:---------------:  |:---------------:|
|**LTAW** |       **1.61**    |       **0.47**   	|        0.92     |


Expected 3px error result of `kitti_finetuned_model.pth.tar` 

Dataset | 3px Error | EPE
:--- | :---: | :---: 
KITTI 2015 training | 0.68 | 0.39
KITTI 2015 testing | 1.81 | N/A

## Acknowledgement
Special thanks to authors of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), [PSMNet](https://github.com/JiaRenChang/PSMNet) and [STTR](https://github.com/mli0603/stereo-transformer#pre-trained-models) for open-sourcing the code.
We also thank GwcNet, CREStereo, AANet for open-sourcing the code. 



