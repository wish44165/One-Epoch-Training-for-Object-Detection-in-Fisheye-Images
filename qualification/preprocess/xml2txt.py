# https://docs.python.org/3/library/xml.etree.elementtree.html

import os
import xml.etree.ElementTree as ET

WIDTH, HEIGHT = 1920, 1080
classList = ['vehicle', 'pedestrian', 'scooter', 'bicycle']

print('====================================================================')

xml_testPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_test/Annotations/All/'
xml_test2Path = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_test2/Annotations/All/'
xml_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2/ivslab_train/Annotations/All/'

txt_testPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_test/'
txt_test2Path = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_test2/'
txt_trainPath = '/home/yuhsi/pro/PAIR-LITEON/data/ivslab_train_v2_txt/ivslab_train/'

xml_testList = os.listdir(xml_testPath)
xml_testList = sorted(xml_testList)
print(len(xml_testList))
for xml_fn in xml_testList:
    xmlPath = xml_testPath + xml_fn
    txtPath = txt_testPath + xml_fn[:-3] + 'txt'
    with open(txtPath, 'w') as f:
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        print('root =', root)
        for child in root:
            for child_i in child:

                # class
                if child_i.tag == 'name':
                    cls_name = child_i.text
                    #print('cls_name =', cls_name)
                    for ci in range(len(classList)):
                        if cls_name == classList[ci]:
                            cls_id = ci
                    print('cls_id =', cls_id)
                
                # bounding box
                if child_i.tag == 'bndbox':
                    for child_ii in child_i:
                        if child_ii.tag == 'xmin':
                            xmin = float(child_ii.text)
                        elif child_ii.tag == 'ymin':
                            ymin = float(child_ii.text)
                        elif child_ii.tag == 'xmax':
                            xmax = float(child_ii.text)
                        else:
                            ymax = float(child_ii.text)
                    print(xmin, ymin, xmax, ymax)
                    cx, cy, w, h = (xmax + xmin) // 2, (ymax + ymin) // 2, xmax - xmin, ymax - ymin
                    xr, yr, wr, hr = cx / WIDTH, cy / HEIGHT, w / WIDTH, h / HEIGHT
                    print('yolo txt =', xr, yr, wr, hr)

                    # write txt
                    info = str(cls_id) + ' ' + str(xr) + ' ' + str(yr) + ' ' + str(wr) + ' ' + str(hr) + '\n'
                    f.write(info)



xml_test2List = os.listdir(xml_test2Path)
xml_test2List = sorted(xml_test2List)
print(len(xml_test2List))
for xml_fn in xml_test2List:
    xmlPath = xml_test2Path + xml_fn
    txtPath = txt_test2Path + xml_fn[:-3] + 'txt'
    with open(txtPath, 'w') as f:
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        print('root =', root)
        for child in root:
            for child_i in child:

                # class
                if child_i.tag == 'name':
                    cls_name = child_i.text
                    #print('cls_name =', cls_name)
                    for ci in range(len(classList)):
                        if cls_name == classList[ci]:
                            cls_id = ci
                    print('cls_id =', cls_id)
                
                # bounding box
                if child_i.tag == 'bndbox':
                    for child_ii in child_i:
                        if child_ii.tag == 'xmin':
                            xmin = float(child_ii.text)
                        elif child_ii.tag == 'ymin':
                            ymin = float(child_ii.text)
                        elif child_ii.tag == 'xmax':
                            xmax = float(child_ii.text)
                        else:
                            ymax = float(child_ii.text)
                    print(xmin, ymin, xmax, ymax)
                    cx, cy, w, h = (xmax + xmin) // 2, (ymax + ymin) // 2, xmax - xmin, ymax - ymin
                    xr, yr, wr, hr = cx / WIDTH, cy / HEIGHT, w / WIDTH, h / HEIGHT
                    print('yolo txt =', xr, yr, wr, hr)

                    # write txt
                    info = str(cls_id) + ' ' + str(xr) + ' ' + str(yr) + ' ' + str(wr) + ' ' + str(hr) + '\n'
                    f.write(info)


xml_trainList = os.listdir(xml_trainPath)
for xml_train_i in xml_trainList:
    xml_train_i_Path = xml_trainPath + xml_train_i
    xml_train_i_List = os.listdir(xml_train_i_Path)
    for xml_fn in xml_train_i_List:
        if xml_fn[-1] == 'l':
            xmlPath = xml_train_i_Path + '/' + xml_fn
            txtPath = txt_trainPath + xml_fn[:-3] + 'txt'
            print(xmlPath)
            print(txtPath)
            with open(txtPath, 'w') as f:
                tree = ET.parse(xmlPath)
                root = tree.getroot()
                print('root =', root)
                for child in root:
                    for child_i in child:

                        # class
                        if child_i.tag == 'name':
                            cls_name = child_i.text
                            #print('cls_name =', cls_name)
                            for ci in range(len(classList)):
                                if cls_name == classList[ci]:
                                    cls_id = ci
                            print('cls_id =', cls_id)
                        
                        # bounding box
                        if child_i.tag == 'bndbox':
                            for child_ii in child_i:
                                if child_ii.tag == 'xmin':
                                    xmin = float(child_ii.text)
                                elif child_ii.tag == 'ymin':
                                    ymin = float(child_ii.text)
                                elif child_ii.tag == 'xmax':
                                    xmax = float(child_ii.text)
                                else:
                                    ymax = float(child_ii.text)
                            print(xmin, ymin, xmax, ymax)
                            cx, cy, w, h = (xmax + xmin) // 2, (ymax + ymin) // 2, xmax - xmin, ymax - ymin
                            xr, yr, wr, hr = cx / WIDTH, cy / HEIGHT, w / WIDTH, h / HEIGHT
                            print('yolo txt =', xr, yr, wr, hr)

                            # write txt
                            info = str(cls_id) + ' ' + str(xr) + ' ' + str(yr) + ' ' + str(wr) + ' ' + str(hr) + '\n'
                            f.write(info)


print(len(os.listdir(txt_testPath)))
print(len(os.listdir(txt_test2Path)))
print(len(os.listdir(txt_trainPath)))