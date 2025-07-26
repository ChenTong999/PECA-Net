# A Hybrid Framework Integrating Paired External and Convolutional Attention for 3D Medical Image Segmentation
Tong Chen, Qinlan Xie, and Xuesong Lu are with the School of Biomedical Engineering, South-Central Minzu University.(Corresponding Author: Xuesong Lu)
***

## Abstract
>The CNN-Transforerm hybrid methods have shown promising performance in medical image segmentation, due to the collaboration of global dependencies and local context. However, the self-attention operation with quadratic complexity models global correlations within individual samples, which is problematic for 3D image segmentation. To address this issue, we propose PECA-Net integrating external attention with convolution for volumetric medical image segmentation. Initially, paired external attention capturing the relationships between different samples is developed by the inter-dependent spatial and channel branches. The sharing queries-keys and different value layers are designed to efficiently learn enriched spatial-channel features, yielding the linear complexity with respect to the input. Additionally, multi-head mechanism built in paired external attention can make all heads interact together. Finally, the spatial and channel attention based on convolution is combined with paired external attention, to provide complementary benefits and serve as the backbone network.

### Architecture overview of PECA-Net
>Overview of PECA-Net approach with hierarchical encoder-decoder structure. The core of the PECA-Net framework is the parallel interaction model (PIM) consisting of paired external attention (PEA) and paired convolutional attention (PCA), which could learn feature representations of global dependencies and local context with efficient computation. The hierarchical encoder comprises four stages, where the resolution of features and the number of channels are respectively decreased and increased by a factor of two in each stage. Concretely, the first stage involves patch embedding layer, followed by our novel PIM. In the patch embedding, the 3D input is partitioned into non-overlapping patches. Then the patches are transformed into channel dimensions, resulting in a high-dimensional tensor. For each remaining stages, a convolutional downsampling layer is employed to decrease the image resolution, followed by the PIM. To fuse global and local features, the PIM consists of two attention blocks (PEA and PCA) that both encode information in spatial and channel dimensions. Instead of self-attention mechanism, the PEA block through external attention captures the correlations of different samples across the whole dataset. The resulting feature maps from the encoder are merged into corresponding feature pyramids of the decoder via skip connections. In each stage of the decoder, the resolution of features and the number of channels are respectively doubled and halved by an upsampling layer using transposed convolution.

<img src="https://github.com/ChenTong999/PECA-Net/raw/master/Architecture overview of PECA-Net.png" width = "800" height = "1600" alt="Architecture" align=center />

### Parameter Count Comparison
>PECA-Net achieves the lower trainable parameter count among comparative frameworks.These results conclusively demonstrate PECA-UNet's parametric efficiency in balancing model complexity and segmentation accuracy.

<img src="https://github.com/ChenTong999/PECA-Net/raw/master/DSC.png" width = "400" height = "400" alt="DSC" align=center />

## Results

### Tumor Dataset
>In the Tumor dataset, the indistinct boundaries of the regions of interest pose a significant challenge for segmentation. In the labeled images, it is evident that the boundaries to be segmented are characterized by irregularity, ambiguity, and the presence of multiple overlapping boundaries. The PECA-UNet’s ability to capture spatial-channel dependencies across the entire dataset significantly enhances its performance in addressing such challenges. 

| Methods | Average | WT | ET | TC |
| :----------: | :-----------:| :----------: | :-----------:| :-----------:|
| TransUNet   | 64.40   | 70.60   | 54.20   | 68.40   |
| U-Net  | 66.40   | 76.60   | 56.10   | 66.50   |
| Att-UNet  | 66.50   | 76.70   | 54.30  | 68.30   |
| CoTr  | 68.30   | 74.60   | 55.70   | 74.80   |
| TransBTS  | 69.60   | 77.90   | 57.40   | 73.50   |
| UNETR  | 71.10   | 78.90   | 58.50   | 76.10   |
| Swin-UNet  | 82.40   | 88.60   | 76.90   | 81.60   |
| nnU-Net  | 84.10   | 91.70   | 78.40   | 82.10   |
| UNETR++  | 84.66   | 91.36   | 78.23   | 84.40   |
| PECA-Net  | 85.85  | 91.74   | 79.83   | 85.97  |
<img src="https://github.com/ChenTong999/PECA-Net/raw/master/PECA_Tumor.png" width = "600" height = "700" alt="Tumor" align=center />

## Installation
### System requirements
We run PECA_Net on a system running Windows 10 with Python 3.6, PyTorch 1.11.0 and CUDA 11.3. The software was tested with the NVIDIA TITAN RTX GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory.

### Installation guide
1. Create and activate conda environment
"""cmd
conda create --name PECA_Net python=3.6
conda activate PECA_Net"""

2. Install PyTorch and torchvision
"""cmd
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"""

3. Install other dependencies
"""cmd
pip install -r requirements.txt"""

### Functions of scripts and folders
1. Evaluation
"""cmd
PECA_Net/inference_acdc.py"""

"""cmd
PECA_Net/inference_synapse.py"""

"""cmd
PECA_Net/inference_tumor.py"""

2. Inference
"""cmd
PECA_Net/inference/predict_simple.py"""

3. Network architecture
"""cmd
PECA_Net/network_architecture/PECA_Net_acdc.py"""

"""cmd
PECA_Net/network_architecture/PECA_Net_synapse.py"""

"""cmd
PECA_Net/network_architecture/PECA_Net_tumor.py"""
***

## Dataset
Dataset can be acquired via following links:

[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
[Synapse](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EbHDhSjkQW5Ak9SMPnGCyb8BOID98wdg3uUvQ0eNvTZ8RA?e=YVhfdg)
[Toumer](http://medicaldecathlon.com/)

The splits of all three datasets are available in dataset_json.

After you have downloaded the datasets, you can follow the settings in [nnUNet]([https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md](https://github.com/MIC-DKFZ/nnUNet/tree/master) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

"""cmd
./PECA_Net/
./DATASET/
  ├── PECA_Net_raw/
      ├── PECA_Net_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── PECA_Net_cropped_data/
  ├── PECA_Net_trained_models/
  ├── PECA_Net_preprocessed/"""

After that, you can preprocess the above data using following commands:

"""cmd
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task01_ACDC
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task03_tumor

nnFormer_plan_and_preprocess -t 1
nnFormer_plan_and_preprocess -t 2
nnFormer_plan_and_preprocess -t 3"""
