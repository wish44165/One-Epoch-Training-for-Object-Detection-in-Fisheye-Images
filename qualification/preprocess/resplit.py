import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

dataPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets/whole/'

dst_txt_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp/train/labels/'
dst_img_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp/train/images/'
dst_txt_valPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp/val/labels/'
dst_img_valPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp/val/images/'


dataList = os.listdir(dataPath)
dataList = sorted(dataList)

# split txt name
nameList = [n for n in dataList if n[-1]=='t']


trainList, valList = train_test_split(nameList, test_size=0.2, random_state=2303113)
print(len(trainList), len(valList))
trainSet, valSet = set(trainList), set(valList)

for txtf in nameList:
    imgf = txtf[:-3] + 'jpg'
    txtPath = dataPath + txtf
    imgPath = dataPath + imgf
    # setup path
    if txtf in trainSet:
        dst_txtPath = dst_txt_trainPath + txtf
        dst_imgPath = dst_img_trainPath + imgf
    else:
        dst_txtPath = dst_txt_valPath + txtf
        dst_imgPath = dst_img_valPath + imgf
    # move file
    shutil.copyfile(txtPath, dst_txtPath)
    shutil.copyfile(imgPath, dst_imgPath)

print(len(os.listdir(dst_txt_trainPath)), len(os.listdir(dst_img_trainPath)), len(os.listdir(dst_txt_valPath)), len(os.listdir(dst_img_valPath)))