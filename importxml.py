#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 下午7:12
# @Author  : cenchaojun
# @File    : importxml.py
# @Software: PyCharm
import xml.etree.ElementTree as ET
import os
import random

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        center_x =  int((obj_struct['bbox'][2] -  obj_struct['bbox'][0])/2) + obj_struct['bbox'][0]
        center_y =  int((obj_struct['bbox'][3] -  obj_struct['bbox'][1])/2) + obj_struct['bbox'][1]
        obj_struct['center'] = [center_x,center_y]
        objects.append(obj_struct)


    return objects

xml_path = '/home/cen/PycharmProjects/dataset/20201208dataset/valdatasetcrop1024xml'
txt_path = '/home/cen/PycharmProjects/dataset/20201208dataset/predict'
for xml in os.listdir(xml_path):
    xml_file = os.path.join(xml_path, xml)
    txt_file = os.path.join(txt_path, xml[:-4] + '.txt')
    ann = parse_rec(filename=xml_file)
    for info in ann:
        print(info)
        print(info['name'])
        conf = random.uniform(a=0.6,b=0.99)
        file = str(info['name']) + ' '+ str(conf)+ ' ' + str(info['bbox'][0]) + ' ' + str(info['bbox'][1]) + ' ' + str(
            info['bbox'][2]) + ' ' + str(info['bbox'][3])
        with open(txt_file,'a') as f:
            # file = str(ann['name']) + ' ' + str(ann['bbox'][0])+ ' ' + str(ann['bbox'][1])+ ' ' + str(ann['bbox'][2])+ ' ' + str(ann['bbox'][3])
            f.write(file + '\n')