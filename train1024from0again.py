#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 上午9:00
# @Author  : cenchaojun
# @File    : train1024from0again.py
# @Software: PyCharm

from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
#  因为增加了augment，所以遮盖batchsize 的大小就变小了
train_dataset = VOCDataset(root_dir='/home/cen/PycharmProjects/dataset/20201203dataset/fewdataset/voc2007',resize_size=[1024,1024],
                           split='1024train',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training").cuda()
# model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('/home/cen/PycharmProjects/FCOS-PyTorch-37.2AP/fewcheckpoint1024/model_60.pth'))

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501

GLOBAL_STEPS = 1
LR_INIT = 2e-4
LR_END = 2e-5
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


model.train()
# 加入这个保存的变量，使得最后能够保存数据
record_pd = pd.DataFrame(columns=['global_step', 'epoch', 'cls_loss', 'cnt_loss', 'reg_loss', 'cost_time', 'lr', 'total_loss'])
for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        #lr = lr_func()
        if GLOBAL_STEPS < WARMPUP_STEPS:
           lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 20001:
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 27001:
           lr = LR_INIT * 0.01
           for param in optimizer.param_groups:
              param['lr'] = lr
        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        #TODO 保存一下模型的输出曲线，保存在一个文件夹中
        print(
            "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
             losses[2].mean(), cost_time, lr, loss.mean()))
        # 明白了tensor转为变量的的方法
        # clc_loss = losses[0].mean().item()
        # cnt_loss = losses[1].mean().item()
        # reg_loss = losses[2].mean().item()
        # total_loss = loss.mean().item()
        # print(clc_loss)
        # print(cnt_loss)
        # print(reg_loss)
        # print(total_loss)
        # record_pd = pd.DataFrame(
        #     columns=['global_step', 'epoch', 'cls_loss', 'cnt_loss', 'reg_loss', 'cost_time', 'lr', 'total_loss'])

        new_row = {'global_step': GLOBAL_STEPS, 'epoch': epoch+1, 'cls_loss': losses[0].mean().item(), 'cnt_loss': losses[1].mean().item(), 'reg_loss': losses[2].mean().item(), 'cost_time':cost_time, 'lr':lr, 'total_loss': loss.mean().item()}
        record_pd = record_pd.append(new_row, ignore_index=True)


        GLOBAL_STEPS += 1
    # 保存模型的参数，
    # 保存整个模型 torch.save(model,"./checkpoint/model_{}.pth".format(epoch + 1))
    torch.save(model.state_dict(),
               "./fewcheckpointfrom0again/model_{}.pth".format(epoch + 1))
record_pd.to_csv('./loss/fewcheckpointfrom0again/loss.csv',index=0)



