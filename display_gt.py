# -*- coding: UTF-8 -*-
"""
Created by root at 2021/4/12
Description:
"""
import torch
import argparse
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import csv
from itertools import islice
import json

def load_pascal_annotation(path):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(path)
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.array([])

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        name = obj.find('name')
        if name.text == 'waterweeds':
            continue
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        boxe = [x1, y1, x2, y2]
        boxes = np.append(boxes, {'name': name.text, 'boxe': boxe})

    # overlaps = scipy.sparse.csr_matrix(overlaps)

    return boxes

def testOne():
    test_img_path = 'test/test.jpg'
    test_label_path = 'test/c000007.txt'
    img = cv2.imread(test_img_path)
    width = img.shape[1]
    height = img.shape[0]
    file1 = open(test_label_path, 'r')
    lines = file1.readlines()
    color_names = ['holothurian','echinus','scallop','starfish']
    colors = {'holothurian': (0, 0, 255), 'echinus': (0, 255, 255), 'scallop': (255, 255, 0), 'starfish': (255, 0, 0)}
    for l in lines:
        ls = l.split("\t")
        cent_x = float(ls[1])*width
        cent_y = float(ls[2])*height
        lebal_width = float(ls[3])*width
        lebal_height = float(ls[4])*height
        x1 = int(cent_x-lebal_width/2)
        x2 = int(cent_x+lebal_width/2)
        y1 = int(cent_y-lebal_height/2)
        y2 = int(cent_y+lebal_height/2)
        obj = color_names[int(ls[0])]
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[obj], 2)
        cv2.putText(img, '{}'.format(obj),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
    cv2.imwrite('test/result.jpg', img)


def testAll():
    imgPath = '/home/louis/Documents/under-water/train/image/'
    gtPath = '/home/louis/Documents/under-water/train/box/'
    savePath = '/home/louis/Documents/under-water/train/image-gt/'
    colors = {'holothurian': (0, 0, 255), 'echinus': (0, 255, 255), 'scallop': (255, 255, 0), 'starfish': (255, 0, 0)}
    input_list = sorted(os.listdir(imgPath))
    num = len(input_list)
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))
        img = cv2.imread(imgPath + input_list[i])
        gt_names = load_pascal_annotation(gtPath + input_list[i][0:-4] + '.xml')
        # print(gt_names)
        # img = img.copy()
        for gt_name in gt_names:
            (x1, y1, x2, y2) = np.array(gt_name['boxe']).astype(np.int)
            obj = gt_name['name']
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[obj], 2)
            cv2.putText(img, '{}'.format(obj),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        cv2.imwrite(savePath + input_list[i], img)

def testPredict():
    test_img_path = '/home/louis/Documents/under-water/URPC_2018/test-A-image'
    test_label_path = 'test/submission.csv'
    colors = {'holothurian': (0, 0, 255), 'echinus': (0, 255, 255), 'scallop': (255, 255, 0), 'starfish': (255, 0, 0)}

    predict={}
    with open(test_label_path, 'r') as f:
        for _line in islice(f, 1, None):
            # print(_line)
            _line = _line.split(',')
            x1, y1, x2, y2 = int(_line[3]), int(_line[4]), int(_line[5]), int(_line[6])
            box=[_line[0],_line[2],x1, y1, x2, y2]
            if _line[1] not in predict.keys():
                predict[_line[1]] = box
            else:
                predict[_line[1]]=np.append(predict[_line[1]],box, axis=0)

    # print(predict)
    for _line in predict:
        print(_line)
        img_name=_line
        img = cv2.imread(f'{test_img_path}/{_line}.jpg')
        _line = predict[_line]
        _line = _line.reshape(int(len(_line)/6),6)
        for _boxs in _line:
            if float(_boxs[1])<0.1:
                break
            x1, y1,x2,  y2 = int(_boxs[2]), int(_boxs[3]), int(_boxs[4]), int(_boxs[5])
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[_boxs[0]], 2)
            cv2.putText(img, '{}'.format(_boxs[1]),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        cv2.imwrite(f'test/predict-gt/{img_name}.jpg', img)

        # reader = csv.reader(f)
        # print(type(reader))
        # for row in reader:
        #     _line = row
        #     print(row)
        #     print(_line)


    # img = cv2.imread(test_img_path)
    # width = img.shape[1]
    # height = img.shape[0]
    # file1 = open(test_label_path, 'r')
    # lines = file1.readlines()
    # color_names = ['holothurian', 'echinus', 'scallop', 'starfish']
    # colors = {'holothurian': (0, 0, 255), 'echinus': (0, 255, 255), 'scallop': (255, 255, 0), 'starfish': (255, 0, 0)}
    # for l in lines:
    #     ls = l.split("\t")
    #     cent_x = float(ls[1]) * width
    #     cent_y = float(ls[2]) * height
    #     lebal_width = float(ls[3]) * width
    #     lebal_height = float(ls[4]) * height
    #     x1 = int(cent_x - lebal_width / 2)
    #     x2 = int(cent_x + lebal_width / 2)
    #     y1 = int(cent_y - lebal_height / 2)
    #     y2 = int(cent_y + lebal_height / 2)
    #     obj = color_names[int(ls[0])]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), colors[obj], 2)
    #     cv2.putText(img, '{}'.format(obj),
    #                 (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (255, 255, 0), 1)
    # cv2.imwrite('test/result.jpg', img)

def load_txt_annotation(test_label_path):
    file1 = open(test_label_path, 'r')
    lines = file1.readlines()
    return lines


def testValidation():
    imgPath = '/home/louis/Documents/under-water/URPC_2018/val_set/images/'
    gtPath = '/home/louis/Documents/under-water/URPC_2018/val_set/labels/'
    savePath = '/home/louis/Documents/under-water/URPC_2018/val_set/image-labels/'
    colors = {'holothurian': (0, 0, 255), 'echinus': (0, 255, 255), 'scallop': (255, 255, 0), 'starfish': (255, 0, 0)}
    color_names = ['holothurian', 'echinus', 'scallop', 'starfish']
    input_list = sorted(os.listdir(imgPath))
    num = len(input_list)
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))
        img = cv2.imread(imgPath + input_list[i])
        width = img.shape[1]
        height = img.shape[0]
        lines = load_txt_annotation(gtPath + input_list[i][0:-4] + '.txt')
        for l in lines:
            ls = l.split("\t")
            cent_x = float(ls[1]) * width
            cent_y = float(ls[2]) * height
            lebal_width = float(ls[3]) * width
            lebal_height = float(ls[4]) * height
            x1 = int(cent_x - lebal_width / 2)
            x2 = int(cent_x + lebal_width / 2)
            y1 = int(cent_y - lebal_height / 2)
            y2 = int(cent_y + lebal_height / 2)
            obj = color_names[int(ls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, '{}'.format(obj),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        cv2.imwrite(f'{savePath}/{input_list[i]}', img)
    csv_path = '/home/louis/Documents/git/yolov5-5.0/runs/test/exp25/labels/submission.csv'
    predict = {}
    with open(csv_path, 'r') as f:
        for _line in islice(f, 1, None):
            # print(_line)
            _line = _line.split(',')
            x1, y1, x2, y2 = int(_line[3]), int(_line[4]), int(_line[5]), int(_line[6])
            box = [_line[0], _line[2], x1, y1, x2, y2]
            if _line[1] not in predict.keys():
                predict[_line[1]] = box
            else:
                predict[_line[1]] = np.append(predict[_line[1]], box, axis=0)

    # print(predict)
    for _line in predict:
        print(_line)
        img_name = _line
        img = cv2.imread(f'{savePath}/{_line}.jpg')
        _line = predict[_line]
        _line = _line.reshape(int(len(_line) / 6), 6)
        for _boxs in _line:
            if float(_boxs[1]) < 0.1:
                break
            x1, y1, x2, y2 = int(_boxs[2]), int(_boxs[3]), int(_boxs[4]), int(_boxs[5])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, '{}'.format(_boxs[1]),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        # cv2.imwrite(f'test/predict-gt/{img_name}.jpg', img)
        cv2.imwrite(f'{savePath}/{img_name}.jpg', img)
    return ''

def val_txt_to_json():
    imgPath = '/home/louis/Documents/under-water/URPC_2018/val_set/images/'
    gtPath = '/home/louis/Documents/under-water/URPC_2018/val_set/labels/'
    anno_json = 'coco/annotations/instances_val2017.json'  # annotations json
    color_names = ['holothurian', 'echinus', 'scallop', 'starfish']
    input_list = sorted(os.listdir(imgPath))
    num = len(input_list)
    jdict=[]
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))
        img = cv2.imread(imgPath + input_list[i])
        width = img.shape[1]
        height = img.shape[0]
        lines = load_txt_annotation(gtPath + input_list[i][0:-4] + '.txt')
        for l in lines:
            ls = l.split("\t")
            cent_x = float(ls[1]) * width
            cent_y = float(ls[2]) * height
            lebal_width = float(ls[3]) * width
            lebal_height = float(ls[4]) * height
            # x1 = int(cent_x - lebal_width / 2)
            # x2 = int(cent_x + lebal_width / 2)
            # y1 = int(cent_y - lebal_height / 2)
            # y2 = int(cent_y + lebal_height / 2)
            b = [cent_x,cent_y, lebal_width,lebal_height]
            # obj = color_names[int(ls[0])]
            jdict.append({'image_id': input_list[i][0:-4],
                          'category_id': int(ls[0]),
                          'bbox': [round(x, 3) for x in b],
                          'score': 1})
    with open(anno_json, 'w') as f:
        json.dump(jdict, f)

def compare_pre_val():
    anno_json = 'coco/annotations/instances_val2017.json'  # annotations json
    pred_json = str( "runs/test/exp31/labels/best_predictions.json")  # predictions json
    print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        # anno = anno.loadRes(anno_json)  # init annotations api
        # pred = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        # if is_coco:
        #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        print(f'map : {map}, ,map50 : {map50}')
    except Exception as e:
        print(f'pycocotools unable to run: {e}')

if __name__ == '__main__':

    # testValidation()
    # val_txt_to_json()
    compare_pre_val()



