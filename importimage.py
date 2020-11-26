#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/24 下午2:12
# @Author  : cenchaojun
# @File    : importimage.py
# @Software: PyCharm
import os
import shutil
from tqdm import tqdm
root = '/home/cen/PycharmProjects/dataset/10m_crop1024_dataset_voc/voc2007'
origin_path = os.path.join(root,'JPEGImages')
save_path = '/home/cen/PycharmProjects/FCOS-PyTorch-37.2AP/testxmlimage'
test_file_path = os.path.join(root,'ImageSets/Main/test.txt')

with open(test_file_path,'r') as f:
    for line in tqdm(f.readlines()):

        line = line.strip('\n') + '.png'
        print(line)
        source_file = os.path.join(origin_path,line)
        tagrt_file = os.path.join(save_path,line)
        shutil.copyfile(source_file,tagrt_file)
