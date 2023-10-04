import os
import numpy as np

################################################################################################################################
#                                                           Summary
################################################################################################################################
################################################################
#                            Official
################################################################
"""
Classes: ['vehicle','pedestrian','scooter','bicycle']
Train / Val / Test: 178004 / 5400 / 5400    (txt and img)
Train: [153928, 497843, 74806, 9690]
Val: [3522, 12856, 994, 25]
Test: [3532, 12638, 1010, 21]
Number of labels: 770865
"""
################################################################
#                            FishEye8K
################################################################
"""
Classes: ['Bus', 'Bike', 'Car', 'Pedestrian', 'Truck']    (bike -> scooter)
Train / Val: 10576 / 5424
Train: [153928, 497843, 74806, 9690, 0]
Val: [3522, 12856, 994, 25, 0]
Number of labels: 157358
"""
################################################################
#                              Valeo
################################################################
"""
Classes: ['vehicles', 'person', 'bicycle']    (real bicycle)
Train / val: 13174 / 3294 
Train: [35464, 12936, 5593]
Val: [8940, 3313, 1460]
Number of labels: 67706
"""
################################################################
#                   Datasets_L (FishEye8K + Valeo)
################################################################
"""
Classes: ['vehicle','pedestrian','scooter','bicycle']
Train / Val: 25974 / 6494
Train: [81285, 22440, 70751, 5652]
Val: [20187, 5568, 17780, 1401]
Number of labels: 225064
"""
################################################################
#        Datasets_f (fisheye effect on official datasets)
################################################################
"""
Classes: ['vehicle','pedestrian','scooter','bicycle']
Train / Val: 178004 / 10800
Train: [497843, 74806, 153928, 9690]
Val: [12856, 994, 3522, 25]
Number of labels: 753664
"""

# dataset path
#OfficalPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets/'    # train / val / test    
#FishEye8KPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fisheye/'    # train / val
#ValeoPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_valeo/'    # train / val
#LPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_L/'    # train / val
#fPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_f/'    # train / val
fpPath = '/home/yuhsi/pro/PAIR-LITEON/data/datasets_fp/'    # train / val

"""
print('################################ Official ################################')
OfficialFolderName = ['train', 'val', 'test']
sss = 0
check_ct = 0
for foldern in OfficialFolderName:
    cls_ct = [0, 0, 0, 0]
    oP = OfficalPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    elif line[0] == '3':
                        cls_ct[3] += 1
                    
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)


print('################################ FishEye8K ################################')
FishEye8KFolderName = ['train', 'val']
sss = 0
check_ct = 0
for foldern in FishEye8KFolderName:
    cls_ct = [0, 0, 0, 0, 0]
    oP = FishEye8KPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    elif line[0] == '3':
                        cls_ct[3] += 1
                    elif line[0] == '4':
                        cls_ct[4] += 1
                    
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)


print('################################ Valeo ################################')
ValeoFolderName = ['train', 'val']
sss = 0
check_ct = 0
for foldern in ValeoFolderName:
    cls_ct = [0, 0, 0]
    oP = ValeoPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)


print('################################ L ################################')
LFolderName = ['train', 'val']
sss = 0
check_ct = 0
for foldern in LFolderName:
    cls_ct = [0, 0, 0, 0]
    oP = LPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    elif line[0] == '3':
                        cls_ct[3] += 1
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)


print('################################ f ################################')
fFolderName = ['train', 'val']
sss = 0
check_ct = 0
for foldern in fFolderName:
    cls_ct = [0, 0, 0, 0]
    oP = fPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    elif line[0] == '3':
                        cls_ct[3] += 1
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)
"""


print('################################ fp ################################')
fpFolderName = ['train', 'val']
sss = 0
check_ct = 0
for foldern in fpFolderName:
    cls_ct = [0, 0, 0, 0]
    oP = fpPath + foldern
    oL = os.listdir(oP)
    for fi in oL:
        fP = oP + '/' + fi
        if fi[-1] == 't':
            with open(fP) as f:
                for line in f.readlines(): 
                    if line[0] == '0':
                        cls_ct[0] += 1
                    elif line[0] == '1':
                        cls_ct[1] += 1
                    elif line[0] == '2':
                        cls_ct[2] += 1
                    elif line[0] == '3':
                        cls_ct[3] += 1
                    check_ct += 1
    print(cls_ct)
    sss += np.sum(cls_ct)
print(sss)
print(check_ct)