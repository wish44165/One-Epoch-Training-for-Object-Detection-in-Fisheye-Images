def txt2pkl(txtPath, imx, imy):
    f = open(txtPath)

    bbox_pkl = []

    for line in f.readlines():
        line = line[:-1]

        # yolo2txt
        line = line.split(' ')
        line_ = [round(float(line[1])*imx - float(line[3])*imx//2), round(float(line[2])*imy - float(line[4])*imy//2), round(float(line[3])*imx), round(float(line[4])*imy)]
        sline = line[0] + ',' + str(line_[0]) + ',' + str(line_[1]) + ',' + str(line_[2]) + ',' + str(line_[3])

        # txt2pkl
        line = sline.split(',')
        line_2 = [np.float64(a) for a in line]
        line_3 = [line_2[1], line_2[2], line_2[1]+line_2[3], line_2[2]+line_2[4]] + [line_2[0]]
        bbox_pkl.append(line_3)
        
    f.close
    bbox_pkl = np.array(bbox_pkl)
    return bbox_pkl

def pkl2txt():
    pass


from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import os
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
#%matplotlib inline


trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp_f/train/'
valPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp_f/val/'

trainList = os.listdir(trainPath)
trainList = sorted(trainList)
txt_trainList = [txtf for txtf in trainList if txtf[-1] == 't']
valList = os.listdir(valPath)
valList = sorted(valList)
txt_valList = [txtf for txtf in valList if txtf[-1] == 't']


# train
for txtf in txt_trainList:
    imgf = txtf[:-3] + 'jpg'
    txtPath = trainPath + txtf
    imgPath = trainPath + imgf

    # load data
    img = cv2.imread(imgPath)
    imy, imx, _ = img.shape
    bboxes = txt2pkl(txtPath, imx, imy)

    # Rotate 1
    img_, bboxes_ = RandomRotate((1, 1))(img.copy(), bboxes.copy())
    cv2.imwrite(trainPath + imgf[:-4] + '_r1.jpg', img_)    # save jpg file

    bbox_pixelList = [bi for bi in bboxes_]    # l: left upper x, left upper y, w, h, cls_id
    with open(trainPath + txtf[:-4] + '_r1.txt', 'w') as f:    # save txt file
        for bbox_pi in bbox_pixelList:
            lux, luy, rbx, rby, cls_id = bbox_pi
            w, h = rbx - lux, rby - luy
            yolox, yoloy, yolow, yoloh = (lux + w/2) / imx, (luy + h/2) / imy, w / imx, h / imy
            string = str(int(cls_id)) + ' ' + str(yolox) + ' ' + str(yoloy) + ' ' + str(yolow) + ' ' + str(yoloh) + '\n'
            f.write(string)

    # Rotate -1
    img_, bboxes_ =RandomRotate((-1, -1))(img.copy(), bboxes.copy())
    cv2.imwrite(trainPath + imgf[:-4] + '_rm1.jpg', img_)    # save jpg file

    bbox_pixelList = [bi for bi in bboxes_]    # l: left upper x, left upper y, w, h, cls_id
    with open(trainPath + txtf[:-4] + '_rm1.txt', 'w') as f:    # save txt file
        for bbox_pi in bbox_pixelList:
            lux, luy, rbx, rby, cls_id = bbox_pi
            w, h = rbx - lux, rby - luy
            yolox, yoloy, yolow, yoloh = (lux + w/2) / imx, (luy + h/2) / imy, w / imx, h / imy
            string = str(int(cls_id)) + ' ' + str(yolox) + ' ' + str(yoloy) + ' ' + str(yolow) + ' ' + str(yoloh) + '\n'
            f.write(string)


# val
for txtf in txt_valList:
    imgf = txtf[:-3] + 'jpg'
    txtPath = valPath + txtf
    imgPath = valPath + imgf

    # load data
    img = cv2.imread(imgPath)
    imy, imx, _ = img.shape
    bboxes = txt2pkl(txtPath, imx, imy)

    # Rotate 1
    img_, bboxes_ = RandomRotate((1, 1))(img.copy(), bboxes.copy())
    cv2.imwrite(valPath + imgf[:-4] + '_r1.jpg', img_)    # save jpg file

    bbox_pixelList = [bi for bi in bboxes_]    # l: left upper x, left upper y, w, h, cls_id
    with open(valPath + txtf[:-4] + '_r1.txt', 'w') as f:    # save txt file
        for bbox_pi in bbox_pixelList:
            lux, luy, rbx, rby, cls_id = bbox_pi
            w, h = rbx - lux, rby - luy
            yolox, yoloy, yolow, yoloh = (lux + w/2) / imx, (luy + h/2) / imy, w / imx, h / imy
            string = str(int(cls_id)) + ' ' + str(yolox) + ' ' + str(yoloy) + ' ' + str(yolow) + ' ' + str(yoloh) + '\n'
            f.write(string)


    # Rotate -1
    img_, bboxes_ = RandomRotate((-1, -1))(img.copy(), bboxes.copy())
    cv2.imwrite(valPath + imgf[:-4] + '_rm1.jpg', img_)    # save jpg file

    bbox_pixelList = [bi for bi in bboxes_]    # l: left upper x, left upper y, w, h, cls_id
    with open(valPath + txtf[:-4] + '_rm1.txt', 'w') as f:    # save txt file
        for bbox_pi in bbox_pixelList:
            lux, luy, rbx, rby, cls_id = bbox_pi
            w, h = rbx - lux, rby - luy
            yolox, yoloy, yolow, yoloh = (lux + w/2) / imx, (luy + h/2) / imy, w / imx, h / imy
            string = str(int(cls_id)) + ' ' + str(yolox) + ' ' + str(yoloy) + ' ' + str(yolow) + ' ' + str(yoloh) + '\n'
            f.write(string)
