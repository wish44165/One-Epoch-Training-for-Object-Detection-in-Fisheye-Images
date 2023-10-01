# [Official YOLOv7](https://github.com/WongKinYiu/yolov7)


## Overview

- [epoch_000.pt (0.5542754)](https://drive.google.com/file/d/187FkcX5Drs3HP_70zw43BXbEKv61-p1U/view?usp=sharing)
- [Folder with Highest score](https://drive.google.com/drive/folders/1wwm0Jx5mC5pu3FLjzhS3ryQwh4PTrofN?usp=sharing)
- [Experimental Results](https://docs.google.com/spreadsheets/d/1FcgC2EaWhQmwmpoyFCBvAWCHqAeutPh-GRuqBahMZfo/edit?usp=sharing)





## Custom Training Details

<details><summary>Conda Envorinment</summary>

```bash
$ conda create -n yolov7 python=3.9 -y
$ conda activate yolov7
```

</details>

<details><summary>Clone Repository</summary>

```bash
$ git clone https://github.com/WongKinYiu/yolov7.git
$ cd yolov7/
$ pip install -r requirements.txt
$ pip install scikit-learn
```

</details>

<details><summary>Demo</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
$ python detect.py --weights yolov7-tiny.pt --source inference/images/horses.jpg --img 640
```

## If Inference without using GPU
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ sudo apt install -y zip htop screen libgl1-mesa-glx
$ pip uninstall torch
$ conda install pytorch torchivision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install torch
```

</details>



<details><summary>custom.yaml</summary>

```bash
cd data/
$ vim custom.yaml
train: /home/yuhsi/pro/PAIR-LITEON/data/datasets/train
val: /home/yuhsi/pro/PAIR-LITEON/data/datasets/val
test: /home/yuhsi/pro/PAIR-LITEON/data/datasets/test
#Classes
nc: 4 # replace according to your number of classes
#classes names
#replace all class names list with your classes names
names: ['vehicle','pedestrian','scooter','bicycle']


```

### Fisheye

```bash
train: /home/yuhsi/pro/PAIR-LITEON/data/datasets_fisheye/train
val: /home/yuhsi/pro/PAIR-LITEON/data/datasets_fisheye/val
#Classes
nc: 5 # replace according to your number of classes
#classes names
#replace all class names list with your classes names
names: ['Bus', 'Bike', 'Car', 'Pedestrian', 'Truck']
```

</details>


### Laptop

<details><summary>train</summary>

```bash
$ python3 train.py --weights yolov7-tiny.pt --data "data/custom.yaml" --workers 16 --batch-size 32 --img 640 --cfg cfg/training/yolov7-tiny.yaml --name yolov7-tiny --hyp data/hyp.scratch.p5.yaml
```

</details>


</details><details><summary>evaluate</summary>

```bash
$ python test.py --data data/custom.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7-tiny/weights/last.pt --name yolov7-tiny
```

</details>


</details><details><summary>inference</summary>

```bash
$ python submit.py --weights ./runs/train/yolov7-tiny/weights/last.pt --conf 0.25 --img-size 640 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


</details><details><summary>inference (4070Ti)</summary>

```bash
$ python submit.py --weights ./runs/train/4070Ti/best.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
# FishEye8K dataset
$ python submit_FishEye8K.py --weights ./runs/train/4070Ti/FishEye8K/finetune/best.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
# FishEye8K + Valeo dataset (dataset_L)
$ python submit.py --weights ./runs/train/4070Ti/stage2/epoch_024.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


### 4070Ti


</details><details><summary>revise loss.py</summary>

- [untimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
#1101](https://github.com/WongKinYiu/yolov7/issues/1101)

If you're training P6 models like e6 or w6 or x, then you'll need to change the following lines as well:

```bash
1389 - matching_matrix = torch.zeros_like(cost) to matching_matrix = torch.zeros_like(cost, device="cpu")
1543 - matching_matrix = torch.zeros_like(cost) to matching_matrix = torch.zeros_like(cost, device="cpu")
```

in the same file (utils/loss.py).

</details>



</details><details><summary>train</summary>

```bash
$ python train_aux.py --weights yolov7-e6e --workers 24 --device 0 --batch-size 2 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e --hyp data/hyp.scratch.p6.yaml
```

</details>

</details><details><summary>inference</summary>

```bash
$ python submit.py --weights ./runs/train/yolov7-tiny/weights/last.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


#### Fisheye

</details><details><summary>fine-tuning (stage 1)</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 2 --data data/custom_fisheye.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-finetune --hyp data/hyp.scratch.p6.yaml
```

</details>


#### Fisheye + Valeo

</details><details><summary>train (stage 2)</summary>

```bash
$ python train_aux.py --weights yolov7-e6e --workers 24 --device 0 --batch-size 2 --data data/custom_L.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-stage2 --hyp data/hyp.scratch.p6.yaml
```

</details>


#### Official_distort

</details><details><summary>train (stage 3)</summary>

- from stage 2 epoch_074.pth

```bash
# mAP so low
$ python train_aux.py --weights runs/train/yolov7-e6e-stage2/weights/epoch_074.pt --workers 24 --device 0 --batch-size 2 --data data/custom_f.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-stage3 --hyp data/hyp.scratch.p6.yaml
# finetune
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 2 --data data/custom_f.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-finetune-stage3 --hyp data/hyp.scratch.p6.yaml
```

</details>

### final stage

</details><details><summary>train (fp_f)</summary>

```bash
# finetune
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 2 --data data/custom_fp_f.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-fp-f --hyp data/hyp.scratch.p6.yaml
# data_aug_2.py
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 2 --data data/custom_fp_f.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-fp-f-r --hyp data/hyp.scratch.p6.yaml
```

</details>


### 2070S


</details><details><summary>fine-tuning</summary>

```bash
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 1 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-finetune --hyp data/hyp.scratch.p6.yaml
```

```bash
$ python train_aux.py --weights ./runs/train/yolov7-e6e-finetune/weights/last.pt --workers 24 --device 0 --batch-size 1 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --hyp data/hyp.scratch.p6.yaml --resume
```

```bash
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 1 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-cheat --hyp data/hyp.scratch.p6.yaml
```

</details>


</details>


<details><summary>fine-tuning (stage 3)</summary>

```bash
$ python train_aux.py --weights yolov7-w6_training.pt --workers 24 --device 0 --batch-size 3 --data data/custom_f.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --name yolov7-w6-finetune-stage3 --hyp data/hyp.scratch.p6.yaml
```

</details>


<details><summary>fine-tuning (final stage)</summary>

```bash
$ python train_aux.py --weights yolov7-e6e_training.pt --workers 24 --device 0 --batch-size 1 --data data/custom_fp.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e-fp --hyp data/hyp.scratch.p6.yaml
```

</details>


</details><details><summary>inference</summary>

```bash
$ python submit.py --weights ./runs/train/yolov7-e6e-finetune/weights/best.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


</details><details><summary>inference (4070Ti)</summary>

```bash
$ python submit.py --weights ./runs/train/yolov7-e6e-epoch1-4070Ti.pt --conf 0.25 --img-size 1280 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


---

### Final Competition

<details><summary>YOLOv7-tiny - train</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
$ python3 train.py --weights yolov7-tiny.pt --data data/custom_fp.yaml --workers 16 --batch-size 48 --img 640 --cfg cfg/training/yolov7-tiny.yaml --name yolov7-tiny --hyp data/hyp.scratch.p5.yaml
```

</details>


<details><summary>YOLOv7 - fine-tuning</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
$ python3 train.py --weights yolov7_training.pt --data data/custom_fp.yaml --workers 16 --batch-size 8 --img 640 --cfg cfg/training/yolov7.yaml --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

</details>


<details><summary>YOLOv7x - fine-tuning</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt
$ python3 train.py --weights yolov7x_training.pt --data data/custom_fp.yaml --workers 16 --batch-size 6 --img 640 --cfg cfg/training/yolov7x.yaml --name yolov7x --hyp data/hyp.scratch.p5.yaml
```

</details>




---





Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/yolov7) using Gradio. Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)

## Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7
```

</details>

## Testing

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69730
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.55521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55937
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.63765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.53766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.73549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83868
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

Multiple GPU training

``` shell
# train p5 models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

Single GPU finetuning for custom dataset

``` shell
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

# finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```

## Re-parameterization

See [reparameterization.ipynb](tools/reparameterization.ipynb)

## Inference

On video:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>


## Export

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

</details>

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113

## Pose estimation

[`code`](https://github.com/WongKinYiu/yolov7/tree/pose) [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

See [keypoint.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/keypoint.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/pose.png" width="39%"/>
    </a>
</div>


## Instance segmentation (with NTU)

[`code`](https://github.com/WongKinYiu/yolov7/tree/mask) [`yolov7-mask.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

See [instance.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/mask.png" width="59%"/>
    </a>
</div>

## Instance segmentation

[`code`](https://github.com/WongKinYiu/yolov7/tree/u7/seg) [`yolov7-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt)

YOLOv7 for instance segmentation (YOLOR + YOLOv5 + YOLACT)

| Model | Test Size | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv7-seg** | 640 | **51.4%** | **69.4%** | **55.8%** | **41.5%** | **65.5%** | **43.7%** |

## Anchor free detection head

[`code`](https://github.com/WongKinYiu/yolov7/tree/u6) [`yolov7-u6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-u6.pt)

YOLOv7 with decoupled TAL head (YOLOR + YOLOv5 + YOLOv6)

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: |
| **YOLOv7-u6** | 640 | **52.6%** | **69.7%** | **57.3%** |


## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```


## Teaser

YOLOv7-semantic & YOLOv7-panoptic & YOLOv7-caption

<div align="center">
    <a href="./">
        <img src="./figure/tennis.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_semantic.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_panoptic.png" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_caption.png" width="24%"/>
    </a>
</div>

YOLOv7-semantic & YOLOv7-detection & YOLOv7-depth (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_city.jpg" width="80%"/>
    </a>
</div>

YOLOv7-3d-detection & YOLOv7-lidar & YOLOv7-road (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_3d.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_lidar.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_road.jpg" width="30%"/>
    </a>
</div>


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>