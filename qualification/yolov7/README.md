## [Official YOLOv7](https://github.com/WongKinYiu/yolov7)



- [YOLOv7-E6E: epoch_000.pt (0.5700583)](https://drive.google.com/file/d/1_8tjqhdgy8UVrWlXnJcFTR4i7erNk0ym/view?usp=sharing)


### YOLOv7-E6E

<details><summary>Create Conda Envorinment</summary>

```bash
$ conda create -n yolov7 python=3.9 -y
$ conda activate yolov7
$ git clone https://github.com/WongKinYiu/yolov7.git
$ cd yolov7/
$ pip install -r requirements.txt
$ pip install scikit-learn
```

### If w/o using GPU
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
nc: 4
names: ['vehicle','pedestrian','scooter','bicycle']
```

</details>


<details><summary>fine-tuning</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt
$ python train_aux.py --weights yolov7-e6e --workers 24 --device 0 --batch-size 2 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --name yolov7-e6e --hyp data/hyp.scratch.p6.yaml
```

</details>


<details><summary>Inference</summary>

```bash
$ python submit.py --weights ./runs/train/yolov7-e6e/epoch_000.pt --conf-thres 0.01 --iou-thres 0.5 --img-size 2176 --source /home/yuhsi/pro/PAIR-LITEON/data/ivslab_test_public --save-txt
```

</details>


### YOLOv7-X

<details><summary>fine-tuning</summary>

```bash
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt
$ python3 train.py --weights yolov7x_training.pt --data data/custom.yaml --workers 16 --batch-size 8 --img 640 --cfg cfg/training/yolov7x.yaml --name yolov7x --hyp data/hyp.scratch.p5.yaml
```

</details>

