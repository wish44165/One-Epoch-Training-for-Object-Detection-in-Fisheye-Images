## [ACM multimedia asia 2023 - Grand Challenge: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](http://www.mmasia2023.org/grand_challenges.html)

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](https://aidea-web.tw/topic/2be7c481-0e16-43b8-8d5d-fb181172144b?focus=intro) (Team name: [yuhsi44165](https://github.com/TW-yuhsi/PAIR-LITEON))

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras(Final Competition)](https://aidea-web.tw/topic/bce44864-7bf6-4a07-a573-fb7ba2c0127a) (Team name: [yuhsi44165](https://github.com/TW-yuhsi/Object-Detection-for-Fisheye-Images-taken-by-Around-view-Cameras))

---



$\large{\textbf{Abstract}}$

This challenge is divided into two stages: qualification and final competition. We will acquire regular image data and need to perform detection on images with a fisheye effect. The approach described in this context begins by taking the original images and transforming them to mimic fisheye effect images for training. Furthermore, this challenge imposes limitations on computational resources, so striking a balance between accuracy and speed is a crucial aspect. In this [paper](https://dl.acm.org/doi/abs/10.1145/3595916.3628349), we asserted that our approach for this competition can achieve high performance with just one epoch of training. In summary, we achieved the top position among 24 participating teams in the qualification competition and secured the fourth position among the 11 successful submitted teams in the final competition.




## 1. Environmental Setup

<details>

<summary>Hardware Information</summary>

- CPU: Intel® Core™ i7-11700F
- GPU: GeForce GTX 1660 SUPER™ VENTUS XS OC (6G)
  
</details>


<details>

<summary>Create Conda Environments</summary>

```bash
$ conda create -n yolov7 python=3.9 -y
$ conda activate yolov7
$ git clone https://github.com/WongKinYiu/yolov7.git
$ cd yolov7/
$ pip install -r requirements.txt
$ pip install scikit-learn
```
  
</details>


<details>

<summary>Create Virtual Environments</summary>

```bash
$ sudo apt-get install python3-virtualenv
$ virtualenv --version    # check version
$ python3.8 -m venv ~/mx
$ . ~/mx/bin/activate
$ pip3 install --upgrade pip wheel
$ pip3 install --extra-index-url https://developer.memryx.com/pip memryx
>> eneural_event
>> memryx23
$ cd ~/mx/
$ pip install -r requirements.txt
```
  
</details>




## 2. Reproducing Details

<details><summary>Dataset Description</summary>

- Download link: details soon

### Restricted classes (a): vehicle→1, pedestrian→2, scooter→3, bicycle→4

| Datasets       | iVS-Dataset                   | FishEye8K                                             | Valeo                                 | Datasets-L (F+V)            |
| -------------- | ----------------------------- | ----------------------------------------------------- | ------------------------------------- | --------------------------- |
| Classes        | (a)                           | Bus:1→1, Bike:2→3, Car:3→1, Pedestrian:4→2, Truck:5→1 | vehicles:1→1, person:2→2, bicycle:3→4 | (a)                         |
| # Train img    | 89002                         | 5288                                                  | 6587                                  | 11875                       |
| # Val img      | 2700                          | 2712                                                  | 1647                                  | 4359                        |
| # Test img     | 2700                          | --                                                    | --                                    | --                          |
| # Train labels | [153928, 497843, 74806, 9690] | [153928, 497843, 74806, 9690, 0]                      | [35464, 12936, 5593]                  | [81285, 22440, 70751, 5652] |
| # Val labels   | [3522, 12856, 994, 25]        | [3522, 12856, 994, 25, 0]                             | [8940, 3313, 1460]                    | [20187, 5568, 17780, 1401]  |
| # Test labels  | [3532, 12638, 1010, 21]       | --                                                    | --                                    | --                          |
| # Total labels | 770865                        | 157358                                                | 67706                                 | 225064                      |

#### Data Augmentation

| Augmented Datasets | Datasets-fp | Datasets-fp-f | Datasets-fp-r | Datasets_fp-r-f |
| ------------------ | ----------- | ------------- | ------------- | --------------- |
| Classes            | (a)         | (a)           | (a)           | (a)             |
| # Train img        | 151042      | 151042        | 453126        | 453126          |
| # Val img          | 37762       | 37762         | 113286        | 113286          |
| # Test img         | --          | --            | --            | --              |
| # Train labels     | [257388, 836476, 122394, 15678] | [257388, 836476, 122394, 15678] | [1029552, 3345904, 489576, 62712] | [1029552, 3345904, 489576, 62712] |
| # Val labels       | [64576, 210198, 31226, 3794]    | [64576, 210198, 31226, 3794]    | [258304, 840792, 124904, 15176]   | [258304, 840792, 124904, 15176]   |
| # Test labels      | --                              | --                              | --                                | --                                |
| # Total labels     | 1541730                         | 1541730                         | 6166920                           | 6166920                           |

</details>

<details>

<summary>Folder Structure on Local Machine</summary>

- Create the following folder structure on the local machine

    ```bash
    # Qualification Competition
    qualification/
    ├── yolov7/
        ├── requirements.txt
        ├── submit.py
        └── test.py
    └── preprocess/
        ├── xml2txt.py
        ├── folderStructure.py
        ├── resplit.py
        ├── fisheye
        ├── data_aug.py
        ├── data_aug_2.py
        └── statistics.py

    # Final Competition
    mx/
    ├── requirements.txt
    ├── calculate.py
    ├── cal_model_size.py
    ├── cal_model_complexity.py
    ├── run_detection_pt.py
    ├── run_detection_onnx.py
    ├── best.csv
    └── best.txt
    ```

</details>


<details><summary>Qualification Competition</summary>

- [Implementation Details](https://github.com/TW-yuhsi/One-Epoch-Training-for-Object-Detection-in-Fisheye-Images/tree/main/qualification)

</details>


<details><summary>Final Competition</summary>

- [Implementation Details](https://github.com/TW-yuhsi/One-Epoch-Training-for-Object-Detection-in-Fisheye-Images/tree/main/mx)

</details>
  



## 3. Demonstration

### 3.1. Comparison between the unaltered image and the fisheye-distorted image

<img src="https://github.com/TW-yuhsi/One-Epoch-Training-for-Object-Detection-in-Fisheye-Images/blob/main/assets/Fig3.jpg" width="70%">


### 3.2. Contrast between artificially fisheye-distorted and fisheye camera-captured image

<img src="https://github.com/TW-yuhsi/One-Epoch-Training-for-Object-Detection-in-Fisheye-Images/blob/main/assets/Fig4.jpg" alt="SwingNet" width="70%" >



## 4. Leaderboard Scores

### 4.1. Qualification Competition

- [epoch_000.pt](https://drive.google.com/file/d/1_8tjqhdgy8UVrWlXnJcFTR4i7erNk0ym/view?usp=sharing)

| Leaderboards     | Filename               | Upload time         | Evaluation result | Ranking |
| ---------------- | ---------------------- | ------------------- | ----------------- | ------- |
| Public & Private | fp-1-0.01-0.5-2172.csv | 2023-08-04 00:51:42 | 0.5700583         | 1/24    |


### 4.2. Final Competition

- [best.pt](https://drive.google.com/file/d/1X7ohtBcc--Ivknpj-NlSlxA2rAF_faPc/view?usp=sharing)

| Team       | Score | Accuracy | Model Complexity GFLOPs | Model size MB | Speed ms | Ranking |
| ---------- | ----- | -------- | ----------------------- | ------------- | -------- | ------- |
| yuhsi44165 | 26.60 | 11.23%   | 195.91                  | 283.34        | 114.80   | 4/11    |




## 5. GitHub Acknowledgement

- [Conversion Tool for FishEye Dataset](https://github.com/leofansq/Tools_KITTI2FishEye)
- [FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection](https://github.com/MoyoG/FishEye8K)
- [WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving](https://github.com/valeoai/WoodScape)
- [Data Augmentation For Object Detection](https://github.com/Paperspace/DataAugmentationForObjectDetection)
- [Official YOLOv7](https://github.com/WongKinYiu/yolov7)




## 6. References

- [Correcting Fisheye Images](https://www.baeldung.com/cs/correcting-fisheye-images)
- [WoodScape: A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving](https://ieeexplore.ieee.org/document/9008254)
- [Visual Chirality](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Visual_Chirality_CVPR_2020_paper.pdf)




## Citation
```
@inproceedings{chen2023one,
  title={One-Epoch Training for Object Detection in Fisheye Images},
  author={Chen, Yu-Hsi},
  booktitle={Proceedings of the 5th ACM International Conference on Multimedia in Asia},
  pages={1--5},
  year={2023}
}
```
