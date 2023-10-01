# Inference for ONNX model
import cv2
import csv
import sys
import time
import requests
import random
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple


cuda = True
w = "./best.onnx"


providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

names = ['vehicle', 'pedestrian', 'scooter', 'bicycle']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


if len(sys.argv) != 3:
    print('USAGE: python run_detection.py [image_list.txt] [filepath of output submission.csv]')
    sys.exit(0)
else:
    inputTXT, outputPath = sys.argv[1], sys.argv[2]
print('image_list.txt =', inputTXT)
print('filepath of output submission.csv =', outputPath)


# read inputTXT
with open(inputTXT) as f:
    lines = f.readlines()
imageList = []
for li in lines:
    if li[-1] != 'g':
        imageList.append(li[:-1])
    else:
        imageList.append(li)
#print(imageList)


with open(outputPath + 'submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_filename", "label_id", "x", "y", "w", "h", "confidence"])

    for p in imageList:
        img = cv2.imread(p)

        imx = img.shape[1]
        imy = img.shape[0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in session.get_outputs()]
        outname

        inname = [i.name for i in session.get_inputs()]
        inname

        inp = {inname[0]:im}

        # ONNX inference
        outputs = session.run(outname, inp)[0]
        #print(outputs)

        ori_images = [img.copy()]

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            #score = round(float(score),3)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)

            #print('box =', box)
            #print('cls_id =', cls_id)
            #print('score =', score)

            name, category, x, y, w, h, conf = p.split('/')[-1], str(cls_id+1), box[0], box[1], int(box[2]-box[0]), int(box[3]-box[1]), score
            preds = [name, category, x, y, w, h, conf]

            if preds[2] < 1:
                preds[2] = 1
            if preds[3] < 1:
                preds[3] = 1
            if preds[4] < 1:
                preds[4] = 1
            if preds[5] < 1:
                preds[5] = 1
            if preds[2] > imx:
                preds[2] = imx
            if preds[3] > imy:
                preds[3] = imy
            if preds[4] > imx:
                preds[4] = imx
            if preds[5] > imy:
                preds[5] = imy
            writer.writerow(preds)

        # display
        #Image.fromarray(ori_images[0]).show()

################################################################################################################################
# USAGE: python run_detection_onnx.py ./imageList.txt Final_example_small/
################################################################################################################################
