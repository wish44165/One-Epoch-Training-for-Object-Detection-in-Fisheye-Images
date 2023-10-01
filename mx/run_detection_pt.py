import argparse
import time
from pathlib import Path

import cv2
import csv
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
#from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = '/home/yuhsi/pro/PAIR-LITEON/stage2/Final_example_small', './best.pt', False, True, 640, not False
    save_img = not False and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    ################################ FINAL ################################
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

        for path in imageList:

            im0s = cv2.imread(path)  # BGR
            assert im0s is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

            # Padded resize
            img = letterbox(im0s, imgsz, stride=32)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    #model(img, augment=opt.augment)[0]
                    model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, 0.01, 0.5, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, ''

                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                    ########
                    # shape
                    ########
                    #print(im0.shape)    # (1080, 1920, 3)
                    imx = im0.shape[1]
                    imy = im0.shape[0]


                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                    ########
                    # name
                    ########
                    pngName = path.split('/')[-1]
                    print(pngName)


                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        lux, luy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()[0:2]
                        
                        preds = [int(cls.item()) + 1] + (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        preds[1:] = [round(preds[i]) for i in range(1,5)]
                        preds[1:3] = [round(lux), round(luy)]
                        if preds[1] < 1:
                            preds[1] = 1
                        if preds[2] < 1:
                            preds[2] = 1
                        if preds[3] < 1:
                            preds[3] = 1
                        if preds[4] < 1:
                            preds[4] = 1
                        if preds[1] > imx:
                            preds[1] = imx
                        if preds[2] > imy:
                            preds[2] = imy
                        if preds[3] > imx:
                            preds[3] = imx
                        if preds[4] > imy:
                            preds[4] = imy
                        preds = [pngName] + preds + [float(conf)]
                        #print(preds)
                        writer.writerow(preds)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':

    global inputTXT, outputPath
    
    if len(sys.argv) != 3:
        print('USAGE: python run_detection.py [image_list.txt] [filepath of output submission.csv]')
        sys.exit(0)
    else:
        inputTXT, outputPath = sys.argv[1], sys.argv[2]
    print('image_list.txt =', inputTXT)
    print('filepath of output submission.csv =', outputPath)

    with torch.no_grad():
        detect()

################################################################################################################################
# USAGE: python run_detection_pt.py ./imageList.txt Final_example_small/
################################################################################################################################
