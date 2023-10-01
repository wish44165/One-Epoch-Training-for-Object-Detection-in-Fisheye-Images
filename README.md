## [Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](http://www.mmasia2023.org/grand_challenges.html)

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras](https://aidea-web.tw/topic/2be7c481-0e16-43b8-8d5d-fb181172144b?focus=intro) (Team name: yuhsi44165)

### [PAIR-LITEON Competition: Embedded AI Object Detection Model Design Contest on Fish-eye Around-view Cameras(Final Competition)](https://aidea-web.tw/topic/bce44864-7bf6-4a07-a573-fb7ba2c0127a) (Team name: yuhsi44165)

---



$\large{\textbf{Abstract}}$

This challenge is divided into two stages: qualification and final competition. We will acquire regular image data and need to perform detection on images with a fisheye effect. The approach outlined in this report involves initially taking the original images and transforming them to resemble images with a fisheye effect for training purposes. Furthermore, this challenge imposes limitations on computational resources, so striking a balance between accuracy and speed is a crucial aspect. To summarize, we secured the first position in the qualification competition and the fourth position in the final competition.




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
  

<details>

<summary>VideoName, ShotSeq, HitFrame</summary>

1. put Badminton/data/part2/test/00170/ .. /00399/ into Badminton/data/part1/val/
    ```bash
    → Badminton/data/part1/val/00001/ .. /00399/    # 1280x720
    # CodaLab
    → Badminton/data/CodaLab/testdata_track1/00170/ .. /00399/    # 1280x720
    ```
2. convert val/+test/ to val_test_xgg/
    ```bash
    $ conda activate golfdb
    $ cd Badminton/src/preprocess/
    $ mkdir val_test_xgg
    $ python3 rt_conversion_datasets.py
    → Badminton/src/preprocess/val_test_xgg/    # 1280x720
    # CodaLab
    → Badminton/src/preprocess/CodaLab/testdata_track1/    # 1280x720
    ```
3. upload val_test_xgg/ to google drive Teaching_Computer_to_Watch_Badminton_Matches_Taiwan_first_competition_combining_AI_and_sports/datasets/part1/
    ```bash
    → Teaching_Computer_to_Watch_Badminton_Matches_Taiwan_first_competition_combining_AI_and_sports/datasets/part1/val_test_xgg/
    → execute golfdb_xgg_inference_best.ipynb
    → src/Notebook/golfdb/golfdb_G3_fold5_iter3000_val_test_X.csv    # 0.0426
    # CodaLab
    → src/Notebook/golfdb/CodaLab_testdata_track1.csv
    ```
  
</details>


<details>

<summary>Hitter</summary>

4. put golfdb_G3_fold5_iter3000_val_test_X.csv into Badminton/src/postprocess/
    ```bash
    → Badminton/src/postprocess/golfdb_G3_fold5_iter3000_val_test_X.csv
    # CodaLab
    → Badminton/src/postprocess/CodaLab/CodaLab_testdata_track1.csv
    ```
5. extract hitframe from csv file
    ```bash
    $ cd Badminton/src/postprocess/
    $ mkdir HitFrame
    $ mkdir HitFrame/1
    $ python3 get_hitframe.py
    >> len(vns), len(hits), len(os.listdir(savePath)) = 4007, 4007, 4007
    → Badminton/src/postprocess/HitFrame/1/    # 720x720, 4007; # CodaLab: 720x720, 2408
    ```
6. execute hitter inference
    ```bash
    $ conda activate ViT_j
    $ cd Badminton/src/ViT-pytorch_Hitter/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_Hitter_ViT-B_16_checkpoint.bin","output/fold2_Hitter_ViT-B_16_checkpoint.bin","output/fold3_Hitter_ViT-B_16_checkpoint.bin","output/fold4_Hitter_ViT-B_16_checkpoint.bin","output/fold5_Hitter_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    → Badminton/src/ViT-pytorch_Hitter/golfdb_G3_fold5_iter3000_val_test_hitter_vote.csv    # 0.0494
    → Badminton/src/ViT-pytorch_Hitter/golfdb_G3_fold5_iter3000_val_test_hitter_mean.csv    # 0.0494
    # CodaLab
    → Badminton/src/ViT-pytorch_Hitter/CodaLab_testdata_track1_hitter_vote.csv
    → Badminton/src/ViT-pytorch_Hitter/CodaLab_testdata_track1_hitter_mean.csv
    ```
  
</details>


<details>

<summary>RoundHead</summary>

7. execute roundhead inference
    ```bash
    $ cd Badminton/src/ViT-pytorch_RoundHead/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_RoundHead_ViT-B_16_checkpoint.bin","output/fold2_RoundHead_ViT-B_16_checkpoint.bin","output/fold3_RoundHead_ViT-B_16_checkpoint.bin","output/fold4_RoundHead_ViT-B_16_checkpoint.bin","output/fold5_RoundHead_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    → Badminton/src/ViT-pytorch_Hitter/golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote.csv    # 0.0527
    → Badminton/src/ViT-pytorch_Hittergolfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean.csv    # 0.0527
    # CodaLab
    → Badminton/src/ViT-pytorch_RoundHead/CodaLab_testdata_track1_hitter_vote_roundhead_vote.csv
    → Badminton/src/ViT-pytorch_RoundHead/CodaLab_testdata_track1_hitter_mean_roundhead_mean.csv
    ```
  
</details>


<details>

<summary>Backhand</summary>

8. execute backhand inference
    ```bash
    $ cd Badminton/src/ViT-pytorch_Backhand/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_Backhand_ViT-B_16_checkpoint.bin","output/fold2_Backhand_ViT-B_16_checkpoint.bin","output/fold3_Backhand_ViT-B_16_checkpoint.bin","output/fold4_Backhand_ViT-B_16_checkpoint.bin","output/fold5_Backhand_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    # CodaLab
    → Badminton/src/ViT-pytorch_Backhand/CodaLab_testdata_track1_hitter_vote_roundhead_vote_backhand_vote.csv
    → Badminton/src/ViT-pytorch_Backhand/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean.csv
    ```
  
</details>


<details>

<summary>BallHeight</summary>

9. execute ballheight inference
    ```bash
    $ cd Badminton/src/ViT-pytorch_BallHeight/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_BallHeight_ViT-B_16_checkpoint.bin","output/fold2_BallHeight_ViT-B_16_checkpoint.bin","output/fold3_BallHeight_ViT-B_16_checkpoint.bin","output/fold4_BallHeight_ViT-B_16_checkpoint.bin","output/fold5_BallHeight_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    # CodaLab
    → Badminton/src/ViT-pytorch_BallHeight/CodaLab_testdata_track1_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote.csv
    → Badminton/src/ViT-pytorch_BallHeight/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean.csv
    ```
  
</details>


<details>

<summary>LandingX</summary>

10. get trajectory
    ```bash
    $ conda activate tracknetv2
    $ cd Badminton/src/TrackNetV2_pytorch/10-10Gray/
    $ mkdir output
    $ python3 predict10_custom.py
    $ mkdir denoise
    $ python3 denoise10_custom.py
    ```
11. execute landingx inference
    ```bash
    $ cd Badminton/src/TrackNetV2_pytorch/10-10Gray/
    $ (mkdir event
    $ cd Badminton/src/TrackNetV2_pytorch/
    $ python3 event_detection_custom.py
    $ python3 HitFrame.py)
    # CodaLab
    → Badminton/src/TrackNetV2_pytorch/CodaLab_tracknetv2_pytorch_10-10Gray_denoise_eventDetection_X.csv
    $ python3 LandingX.py
    # CodaLab
    → Badminton/src/TrackNetV2_pytorch/CodaLab_testdata_track1_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LXY.csv
    → Badminton/src/TrackNetV2_pytorch/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LXY.csv
    ```
  
</details>


<details>

<summary>LandingY, HitterLocationX, HitterLocationY, DefenderLocationX, DefenderLocationY</summary>

12. extract hitframe for yolo from csv
    ```bash
    $ cd Badminton/src/postprocess/
    $ mkdir HitFrame_yolo
    $ python3 get_hitframe_yolo.py
    → Badminton/src/postprocess/HitFrame_yolo/    # 1280x720, 4007; CodaLab: 1280x720, 2408
    ```
13. execute yolov5 inference
    ```bash
    $ conda activate yolov5
    $ cd Badminton/src/yolov5/
    $ python3 detect.py --weights runs/train/exp/weights/best.pt --source /home/yuhsi/Badminton/src/postprocess/HitFrame_yolo/ --conf-thres 0.3 --iou-thres 0.3 --save-txt --imgsz 2880 --agnostic-nms --augment
    → Badminton/src/yolov5/runs/detect/exp/    # 4007
    # CodaLab
    $ python3 detect.py --weights runs/train/exp/weights/best.pt --source /home/yuhsi/Badminton/src/postprocess/HitFrame_yolo/ --conf-thres 0.3 --iou-thres 0.3 --save-txt --imgsz 2880 --agnostic-nms --augment
    → Badminton/src/yolov5/runs/detect/exp2/    # 2408
    ## video demo
    $ python3 detect.py --weights runs/train/exp/weights/best.pt --source /home/yuhsi/Badminton/data/CodaLab/testdata_track1/00171/00171.mp4 --conf-thres 0.3 --iou-thres 0.3 --save-txt --imgsz 2880 --agnostic-nms --augment
    $ python3 demo.py
    ```
14. execute landingy inference
    ```bash
    $ mkdir runs/detect/exp_draw
    $ mkdir runs/detect/exp_draw/case1
    $ python3 LandingY_Hitter_Defender_Location.py
    ```
  
</details>


<details>

<summary>BallType</summary>

15. execute balltype inference
    ```bash
    $ conda activate ViT_j
    $ cd Badminton/src/ViT-pytorch_BallType/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_BallType_ViT-B_16_checkpoint.bin","output/fold2_BallType_ViT-B_16_checkpoint.bin","output/fold3_BallType_ViT-B_16_checkpoint.bin","output/fold4_BallType_ViT-B_16_checkpoint.bin","output/fold5_BallType_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    # CodaLab
    → Badminton/src/ViT-pytorch_BallType/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote.csv
    → Badminton/src/ViT-pytorch_BallType/CodaLab_testdata_track1_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LX_LY_case1_HD_balltype_mean.csv
    ```
  
</details>


<details>

<summary>Winner</summary>

16. execute winner inference
    ```bash
    $ cd Badminton/src/Vit-pytorch_Winner/
    $ python3 submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/fold1_Winner_ViT-B_16_checkpoint.bin","output/fold2_Winner_ViT-B_16_checkpoint.bin","output/fold3_Winner_ViT-B_16_checkpoint.bin","output/fold4_Winner_ViT-B_16_checkpoint.bin","output/fold5_Winner_ViT-B_16_checkpoint.bin"] --img_size [480,480,480,480,480]
    # CodaLab
    → Badminton/src/ViT-pytorch_Winner/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case1.csv
    ```
  
</details>


<details>

<summary>HitterLocationX, HitterLocationY, DefenderLocationX, DefenderLocationY (Updated)</summary>

17. use yolov8x-pose-p6.pt model to execute pose estimation
    ```bash
    $ cd Badminton/src/ultralytics/
    $ mkdir pose_estimation
    $ python3 submit.py
    → Badminton/src/ViT-pytorch_Winner/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case1_v8pose.csv
    ## video demo
    $ python3 demo.py
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

### 4.1. AICUP2023
|Leaderboards | Filename                                                                                        | Upload time | Evaluation result | Ranking |
|----| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- |
|Public | [golfdb_G3_fold5_...csv](https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/submit/golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case2.csv) | 2023-05-15 22:21:17                   | 0.0727                 | 11/30                  |
|Private | [golfdb_G3_fold5_...csv](https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/submit/golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case2.csv) | 2023-05-15 22:21:17                   | 0.0622                 | 11/30                  |

### 4.2. CodaLab2023
|Leaderboards | Filename                                                                                        | Upload time | Evaluation result | Ranking |
|----| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- |
|Final phase | [CodaLab_testdata_track1_...csv](https://github.com/TW-yuhsi/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/blob/main/submit/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case1_v8pose.csv) | 2023-06-17 16:03                   | Panding                 | Panding                  |




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
