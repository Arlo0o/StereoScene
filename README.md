# StereoScene[IJCAI2024]
This repository contains the official implementation of the IJCAI2024 paper: "Bridging Stereo Geometry and BEV Representation with Reliable Mutual Interaction for Semantic Scene Completion".

# Teaser
- **Comparison with MonoScene on SemanticKITTI:**
<p align="center">
<img src="./teaser/demo.gif" />
</p>


- **Quantitative Results:**
<p align="center">
<img src="./teaser/results.png" />
</p>

# Table of Content
- [News](#news)
- [Abstract](#abstract)
- [Installation](#step-by-step-installation-instructions)
- [Prepare Data](#prepare-data)
- [Pretrained Model](#pretrained-model)
- [Training & Evaluation](#training--evaluation)
- [Visualization](#visualization)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


# News
- [2023/03]: Paper is on [arxiv](https://arxiv.org/abs/2303.13959)
- [2023/03]: Demo and code released.
- [2024/04]: Paper is accepted on IJCAI 2024.
- [2025/01]: Update visualization tools.

# Abstract
3D semantic scene completion (SSC) is an ill-posed perception task that requires inferring a dense 3D scene from limited observations. Previous camera-based methods struggle to predict accurate semantic scenes due to inherent geometric ambiguity and incomplete observations. In this paper, we resort to stereo matching technique and bird's-eye-view (BEV) representation learning to address such issues in SSC. Complementary to each other, stereo matching mitigates geometric ambiguity with epipolar constraint while BEV representation enhances the hallucination ability for invisible regions with global semantic context. However, due to the inherent representation gap between stereo geometry and BEV features, it is non-trivial to bridge them for dense prediction task of SSC. Therefore, we further develop a unified occupancy-based framework dubbed BRGScene, which effectively bridges these two representations with dense 3D volumes for reliable semantic scene completion. Specifically, we design a novel Mutual Interactive Ensemble (MIE) block for pixel-level reliable aggregation of stereo geometry and BEV features. Within the MIE block, a Bi-directional Reliable Interaction (BRI) module, enhanced with confidence re-weighting, is employed to encourage fine-grained interaction through mutual guidance. Besides, a Dual Volume Ensemble (DVE) module is introduced to facilitate complementary aggregation through channel-wise recalibration and multi-group voting. Our method outperforms all published camera-based methods on SemanticKITTI for semantic scene completion. 

# Step-by-step Installation Instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
python > 3.7 may not be supported, because installing open3d-python with py>3.7 causes errors.
```shell
conda create -n occupancy python=3.7 -y
conda activate occupancy
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**c. Install gcc>=5 in conda env (optional).**
I do not use this step.
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
Please check your CUDA version for [mmdet3d](https://github.com/open-mmlab/mmdetection3d/issues/2427) if encountered import problem. 

**f. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
pip install PyMCubes
```


## Known problems

### AttributeError: module 'distutils' has no attribute 'version'
The error appears due to the version of "setuptools", try:
```shell
pip install setuptools==59.5.0
```




# Prepare Data

- **a. You need to download**

     - The **Odometry calibration** (Download odometry data set (calibration files)) and the **RGB images** (Download odometry data set (color)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), extract them to the folder `data/occupancy/semanticKITTI/RGB/`.
     - The **Velodyne point clouds** (Download [data_odometry_velodyne](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)) and the **SemanticKITTI label data** (Download [data_odometry_labels](http://www.semantic-kitti.org/assets/data_odometry_labels.zip)) for sparse LIDAR supervision in training process, extract them to the folders ``` data/lidar/velodyne/ ``` and ``` data/lidar/lidarseg/ ```, separately. 


- **b. Prepare KITTI voxel label (see sh file for more details)**
```
bash process_kitti.sh
```


# Pretrained Model

Download [StereoScene pretrained model](https://drive.google.com/file/d/1D0gP3S5uKo6pDZApCg7lrwOf5c5_yvC7/view?usp=share_link) on SemanticKITTI and [Efficientnet-b7 pretrained model](https://drive.google.com/file/d/1JffT44Zjw27XBTeUv8_RW6wP6GllMtZh/view?usp=share_link), put them in the folder `/pretrain`.



# Training & Evaluation

## Single GPU
- **Train with single GPU:**
```
export PYTHONPATH="."  
python tools/train.py   \
            projects/configs/occupancy/semantickitti/stereoscene.py
```

- **Evaluate with single GPUs:**
```
export PYTHONPATH="."  
python tools/test.py  \
            projects/configs/occupancy/semantickitti/stereoscene.py \
            pretrain/pretrain_stereoscene.pth 
```


## Multiple GPUS
- **Train with n GPUs:**
```
bash run.sh  \
        projects/configs/occupancy/semantickitti/stereoscene.py n
```

- **Evaluate with n GPUs:**
```
 bash tools/dist_test.sh  \
            projects/configs/occupancy/semantickitti/stereoscene.py \
            pretrain/pretrain_stereoscene.pth  n
```


## Visualization

We use mayavi to visualize the predictions. Please install [mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html) following the official installation instruction. Then, use the following commands to visualize the outputs.


```
export PYTHONPATH="."  
python tools/save_vis.py projects/configs/occupancy/semantickitti/stereoscene.py \
            pretrain/pretrain_stereoscene.pth  --eval mAP
python tools/visualization.py
```


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Citation
If you find this project useful in your research, please consider cite:
```
@misc{li2023stereoscene,
      title={StereoScene: BEV-Assisted Stereo Matching Empowers 3D Semantic Scene Completion}, 
      author={Bohan Li and Yasheng Sun and Xin Jin and Wenjun Zeng and Zheng Zhu and Xiaoefeng Wang and Yunpeng Zhang and James Okae and Hang Xiao and Dalong Du},
      year={2023},
      eprint={2303.13959},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgements
Many thanks to these excellent open source projects: 
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [SSC](https://github.com/waterljwant/SSC)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [Semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) 
- [Pseudo_Lidar_V2](https://github.com/mileyan/Pseudo_Lidar_V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

