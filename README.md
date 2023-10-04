## [Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](http://www.mmasia2023.org/grand_challenges.html)

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](https://aidea-web.tw/topic/2be7c481-0e16-43b8-8d5d-fb181172144b?focus=intro) (Team name: yuhsi44165)

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras(Final Competition)](https://aidea-web.tw/topic/bce44864-7bf6-4a07-a573-fb7ba2c0127a) (Team name: yuhsi44165)

---



$\large{\textbf{Abstract}}$

This challenge is divided into two stages: qualification and final competition. We will acquire regular image data and need to perform detection on images with a fisheye effect. The approach described in this context begins by taking the original images and transforming them to mimic fisheye effect images for training. Furthermore, this challenge imposes limitations on computational resources, so striking a balance between accuracy and speed is a crucial aspect. In this paper, we asserted that our approach for this competition can achieve high performance with just one epoch of training. In summary, we achieved the top position among 24 participating teams in the qualification competition and secured the fourth position among the 11 successful submitted teams in the final competition.




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




## 2. Inference Details

<details><summary>Dataset Description</summary>

### Restricted classes (a): vehicle→1, pedestrian→2, scooter→3, bicycl→4

| Datasets       | iVS-Dataset                   | FishEye8K                                             | Valeo                                 | Datasets_L (F+V)            |
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

| Augmented Datasets | Datasets_fp | Datasets_fp-f | Datasets_fp-r | Datasets_fp-f-r |
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
        └── submit.py
    └── preprocess/
        ├── DataAugmentationForObjectDetection/
        ├── Tools_KITTI2FishEye/
        └── resplit.py
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
  



## 3. Demonstration

### 3.1. Optical Flow Calculation embedded in Reynolds Transport Theorem

[<img src="https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/assets/Fig1.jpg" width="70%">](https://youtu.be/6Lm6zaKWwhk)


### 3.2. SwingNet (MobileNetV2 + bidirectional LSTM)

<img src="https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/assets/Fig2.jpg" alt="SwingNet" width="70%" >


### 3.3. YOLOv5 & TrackNetV2 & YOLOv8-pose 

[<img src="https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/assets/Fig6.jpg" width="70%">](https://youtu.be/Bkc9bswT5uE)



## 4. Leaderboard Scores

### 4.1. Qualification Competition
| Leaderboards     | Filename               | Upload time         | Evaluation result | Ranking |
| ---------------- | ---------------------- | ------------------- | ----------------- | ------- |
| Public & Private | fp-1-0.01-0.5-2172.csv | 2023-08-04 00:51:42 | 0.5700583         | 1/24    |


### 4.2. Final Competition
| Team       | Score | Accuracy | Model Complexity GFLOPs | Model size MB | Speed ms | Ranking |
| ---------- | ----- | -------- | ----------------------- | ------------- | -------- | ------- |
| yuhsi44165 | 26.60 | 11.23%   | 195.91                  | 283.34        | 114.80   | 4/11    |



## 5. GitHub Acknowledgement

- [TrackNetV2: N-in-N-out Pytorch version (GitLab)](https://nol.cs.nctu.edu.tw:234/lukelin/TrackNetV2_pytorch)
- [GolfDB: A Video Database for Golf Swing Sequencing](https://github.com/wmcnally/golfdb)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://github.com/jeonsworld/ViT-pytorch)
- [Ultralytics YOLOv5 v7.0](https://github.com/ultralytics/yolov5)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [A PyTorch implementation of MobileNetV3](https://github.com/xiaolai-sqlai/mobilenetv3)
- [Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN)](https://github.com/locuslab/TCN)




## 6. References

- [Lucas/Kanade meets Horn/Schunck: Combining local and global optic flow methods](http://helios.mi.parisdescartes.fr/~lomn/Cours/CV/SeqVideo/Articles2017/ShunckMeetsKanade_4.pdf)
- [Stochastic representation of the Reynolds transport theorem: revisiting large-scale modeling](https://arxiv.org/pdf/1611.03413)
- [TrackNetV2: Efficient shuttlecock tracking network](https://ieeexplore.ieee.org/iel7/9302522/9302594/09302757.pdf)
- [GolfDB: A Video Database for Golf Swing Sequencing](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/McNally_GolfDB_A_Video_Database_for_Golf_Swing_Sequencing_CVPRW_2019_paper.pdf)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [Ultralytics YOLOv5 v7.0](https://ui.adsabs.harvard.edu/abs/2022zndo...7347926J/abstract)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/modes/predict/#working-with-results)
- [Searching for mobilenetv3](http://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf%E3%80%82%E6%9C%AC%E6%96%87%E5%BC%95%E7%94%A8%E7%94%A8(%5C*)%E8%A1%A8%E7%A4%BA%E3%80%82)




## Citation
```
@misc{chen2023new,
      title={A New Perspective for Shuttlecock Hitting Event Detection}, 
      author={Yu-Hsi Chen},
      year={2023},
      eprint={2306.10293},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
