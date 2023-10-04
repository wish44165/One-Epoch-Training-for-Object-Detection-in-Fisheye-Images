## [Official YOLOv7](https://github.com/WongKinYiu/yolov7)


## Overview


- [YOLOv7-E6E: epoch_000.pt (0.5700583)](https://drive.google.com/file/d/187FkcX5Drs3HP_70zw43BXbEKv61-p1U/view?usp=drive_link)

- [epoch_000.pt (0.5542754)](https://drive.google.com/file/d/187FkcX5Drs3HP_70zw43BXbEKv61-p1U/view?usp=sharing)
- [Folder with Highest score](https://drive.google.com/drive/folders/1wwm0Jx5mC5pu3FLjzhS3ryQwh4PTrofN?usp=sharing)





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

<details><summary>Train</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt
$ python3 train.py --weights yolov7x_training.pt --data data/custom_fp.yaml --workers 16 --batch-size 6 --img 640 --cfg cfg/training/yolov7x.yaml --name yolov7x --hyp data/hyp.scratch.p5.yaml
```

## If without using GPU
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




### Final Competition



<details><summary>YOLOv7-X: fine-tuning</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt
$ python3 train.py --weights yolov7x_training.pt --data data/custom_fp.yaml --workers 16 --batch-size 8 --img 640 --cfg cfg/training/yolov7x.yaml --name yolov7x --hyp data/hyp.scratch.p5.yaml
```

</details>

