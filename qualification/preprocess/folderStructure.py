import os
import shutil

txt_testPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_test/'    # val
txt_testList = os.listdir(txt_testPath)
txt_test2Path = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_test2/'    # test
txt_test2List = os.listdir(txt_test2Path)
txt_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_train/'    # train
txt_trainList = os.listdir(txt_trainPath)

valPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets/val/'
testPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets/test/'
trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets/train/'

# copy txt
for txt_i in txt_testList:
    shutil.copyfile(txt_testPath + txt_i, valPath + txt_i)

for txt_i in txt_test2List:
    shutil.copyfile(txt_test2Path + txt_i, testPath + txt_i)

for txt_i in txt_trainList:
    shutil.copyfile(txt_trainPath + txt_i, trainPath + txt_i)

# copy jpg
jpg_testPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_test/JPEGImages/All/'
jpg_testList = os.listdir(jpg_testPath)
jpg_test2Path = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_test2/JPEGImages/All/'
jpg_test2List = os.listdir(jpg_test2Path)
jpg_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_train/JPEGImages/All/'
jpg_trainList = os.listdir(jpg_trainPath)

for jpg_i in jpg_testList:
    shutil.copyfile(jpg_testPath + jpg_i, valPath + jpg_i)

for jpg_i in jpg_test2List:
    shutil.copyfile(jpg_test2Path + jpg_i, testPath + jpg_i)

for train_i in jpg_trainList:
    jpg_train_i_Path = jpg_trainPath + train_i
    jpg_train_i_List = os.listdir(jpg_train_i_Path)
    for jpg_i in jpg_train_i_List:
        jpgPath = jpg_train_i_Path + '/' + jpg_i
        shutil.copyfile(jpgPath, trainPath + jpg_i)

print(len(os.listdir(valPath)))
print(len(os.listdir(testPath)))
print(len(os.listdir(trainPath)))