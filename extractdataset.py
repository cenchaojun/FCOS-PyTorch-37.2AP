#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 下午9:29
# @Author  : cenchaojun
# @File    : extractdataset.py
# @Software: PyCharm
import os
import glob
import random
import shutil

trained_path = '/home/cen/PycharmProjects/dataset/20201203dataset/fewtraindataset'
trained_file = glob.glob(trained_path + '/*.JPG')
image_file = set([image.split('/')[-1] for image in trained_file ])
total_path = '/home/cen/PycharmProjects/dataset/20201203dataset/traindataset'
total_file = glob.glob(total_path + '/*.JPG')
total_image_file = set([image.split('/')[-1] for image in total_file ])
reset = list(total_image_file - image_file)
total_num = len(total_file)
extract_num = int(total_num * 0.1)
extract_image = random.sample(reset,extract_num)
extract_path = '/home/cen/PycharmProjects/dataset/20201203dataset/extractdataset'
for image in extract_image:
    image_path = os.path.join(total_path,image)
    save_path = os.path.join(extract_path,image)
    shutil.copyfile(src=image_path,dst=save_path)



