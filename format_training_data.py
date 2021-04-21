# -*- coding: UTF-8 -*-
"""
Created by root at 2021/4/12
Description:
"""
import torch
import argparse
import os
import numpy as np
from shutil import copyfile, rmtree
import json
import xml.etree.ElementTree as ET
from tqdm.autonotebook import tqdm


def load_pascal_annotation(path):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(path)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    size = tree.findall('size')
    width = int(size[0].find('width').text)
    height = int(size[0].find('height').text)
    size = [width, height]

    # num_objs = len(objs)

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
        boxes = np.append(boxes, {'name': name.text, 'boxe': boxe, 'size': size})

    # overlaps = scipy.sparse.csr_matrix(overlaps)

    return boxes, size, tree.findall('frame')[0].text


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    img_path = '/home/louis/Documents/under-water/URPC_2018/train/image/'
    box_path = '/home/louis/Documents/under-water/URPC_2018/train/box/'
    annotations_path = '/home/louis/Documents/under-water/URPC_2018/annotations/'
    train_set_path = '/home/louis/Documents/under-water/URPC_2018/train_set/'
    train_set_images = train_set_path + 'images/'
    train_set_labels = train_set_path + 'labels/'
    val_set_path = '/home/louis/Documents/under-water/URPC_2018/val_set/'
    val_set_images = val_set_path + 'images/'
    val_set_labels = val_set_path + 'labels/'
    rmtree(train_set_path)
    rmtree(val_set_path)
    # rmtree(annotations_path)
    os.makedirs(os.path.dirname(train_set_images), exist_ok=True)
    os.makedirs(os.path.dirname(train_set_labels), exist_ok=True)
    os.makedirs(os.path.dirname(val_set_images), exist_ok=True)
    os.makedirs(os.path.dirname(val_set_labels), exist_ok=True)
    input_list = sorted(os.listdir(img_path))
    num = len(input_list)
    val_set_num = np.random.choice(num, int(num / 50), replace=False)
    train_set_num = [k for k in range(num) if k not in val_set_num]

    progress_bar = tqdm(val_set_num)
    class_tag = ['holothurian', 'echinus', 'scallop', 'starfish']
    for i in progress_bar:
        progress_bar.update()

        _annotation, size, file_name_simple = load_pascal_annotation(box_path + input_list[i][0:-4] + '.xml')
        _width = size[0]
        _height = size[1]
        _file_name = input_list[i]
        _this_annotations = []
        if len(_annotation) != 0:
            for gt_name in _annotation:
                (x1, y1, x2, y2) = np.array(gt_name['boxe']).astype(np.int)
                _class_tag = class_tag.index(gt_name['name'])
                center_x, center_y, __width, __height = 0.5 * (x2 + x1) / _width, 0.5 * (y2 + y1) / _height, (
                        x2 - x1) / _width, (y2 - y1) / _height
                _this_annotations = np.append(_this_annotations,
                                              f"{_class_tag}\t{center_x}\t{center_y}\t{__width}\t{__height}")
        if len(_this_annotations) > 0:
            copyfile(img_path + input_list[i], val_set_images + input_list[i])
            with open(val_set_labels + file_name_simple + '.txt', 'a') as p:
                for i in _this_annotations:
                    p.write(i)
                    p.write('\n')

    progress_bar = tqdm(train_set_num)
    for i in progress_bar:
        progress_bar.update()

        _annotation, size, file_name_simple = load_pascal_annotation(box_path + input_list[i][0:-4] + '.xml')
        _width = size[0]
        _height = size[1]
        _file_name = input_list[i]
        _this_annotations = []
        if len(_annotation) != 0:
            for gt_name in _annotation:
                (x1, y1, x2, y2) = np.array(gt_name['boxe']).astype(np.int)
                _class_tag = class_tag.index(gt_name['name'])
                center_x, center_y, __width, __height = 0.5 * (x2 + x1) / _width, 0.5 * (y2 + y1) / _height, (
                        x2 - x1) / _width, (y2 - y1) / _height
                _this_annotations = np.append(_this_annotations,
                                              f"{_class_tag}\t{center_x}\t{center_y}\t{__width}\t{__height}")
        if len(_this_annotations) > 0:
            copyfile(img_path + input_list[i], train_set_images + input_list[i])
            with open(train_set_labels + file_name_simple + '.txt', 'a') as p:
                for i in _this_annotations:
                    p.write(i)
                    p.write('\n')
