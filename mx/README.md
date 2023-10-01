# YOLOv7-X

## Folder Structure

```bash
YOLOv7/
├── model_0_best.onnx
├── model_0_best_post.onnx
├── requirements.txt
├── calculate.py
├── cal_model_size.pdf
├── cal_model_size.py
├── cal_model_complexity.pdf
├── cal_model_complexity.py
├── run_detection_pt.py
├── run_detection_onnx.py
├── best.dfp
├── best.onnx
├── best.pt
├── best.csv
├── best.txt
└── techreport.pdf
```


<details>

<summary>0.1.Create Conda Environments</summary>

```bash
$ conda create -n yolov7 python=3.9 -y
$ conda activate yolov7
$ pip install -r requirements.txt
```
  
</details>


<details><summary>0.2.Setup for Weight Compilation</summary>

```bash
# 1.
$ conda deactivate
$ sudo apt-get install python3-virtualenv
$ virtualenv --version    # check version
# 2.
$ python3.8 -m venv ~/mx    # fp-r
$ . ~/mx/bin/activate

$ pip3 install --upgrade pip wheel
# 3.
$ pip3 install --extra-index-url https://developer.memryx.com/pip memryx
>> eneural_event
>> memryx23
# install packages
$ cd mx/    # fp-r

$ git clone https://github.com/WongKinYiu/yolov7.git
$ cd yolov7/
$ pip install -r requirements.txt
```

</details>



<details><summary>1.1.Convert the model format to ONNX file format</summary>

```bash
# python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
#        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

# x
$ python export.py --weights best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.5 --conf-thres 0.001 --img-size 640 640 --max-wh 640
```

- https://github.com/lutzroeder/netron

</details>


<details><summary>1.2.Compile the model and crop the unsupported operation</summary>

```bash
# x
$ mx_nc -vvv -m best.onnx --outputs onnx::Reshape_575,onnx::Reshape_609,onnx::Reshape_643 -g 3.1 -c 8 | tee compile_log.txt
```

</details>


<details><summary>1.3.Test the benchmark of the compiled and cropped model to obtain the final performance</summary>

```bash
# x
$ mx_sim -v -d best.dfp
```

</details>


<details><summary>2.1.calculate.py</summary>

```bash
$ python calculate.py
```

</details>


<details><summary>2.2.run_detection.py</summary>


### run_detection_pt.py

```bash
$ python run_detection_pt.py ./imageList.txt Final_example_small/
```

### run_detection_onnx.py

```bash
$ python run_detection_onnx.py ./imageList.txt Final_example_small/
```

</details>


<details><summary>2.3.cal_model_size.py</summary>

```bash
$ python cal_model_size.py
```

</details>


<details><summary>2.4.cal_model_complexity.py</summary>

- https://github.com/ThanatosShinji/onnx-tool
- https://github.com/gmalivenko/onnx-opcounter

```bash
$ python cal_model_complexity.py
```

</details>