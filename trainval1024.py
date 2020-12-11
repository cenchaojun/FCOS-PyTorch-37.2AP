#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 上午10:08
# @Author  : cenchaojun
# @File    : trainval1024.py
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
import torch
import numpy as np
import cv2
from tqdm import tqdm
from collections import OrderedDict, Counter
import pandas as pd


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores


def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    # 得到一些数值，将前面的0和后面的1加到这个数组中去
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    # 用的也是插值法。好像是第二种吧
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_2d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # 试一下没有排序的方式
        # fp_nosort = fp
        # tp_nosort = tp
        # fp_nosort = np.cumsum(fp_nosort)
        # tp_nosort = np.cumsum(tp_nosort)
        # recall_nosort = tp_nosort / total_gts
        # precision_nosort = tp_nosort / np.maximum(tp_nosort + fp_nosort, np.finfo(np.float64).eps)
        # print("precision_nosort: {}".format(precision_nosort))
        # print("recall_nosort: {}".format(recall_nosort))
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        fp_num = dict(Counter(fp).items())[1.0]
        print("fp counter : {0}".format(fp_num))
        tp_num = dict(Counter(tp).items())[1.0]
        print("tp counter : {0}".format(tp_num))
        precision_value = tp_num / (tp_num + fp_num)
        recall_value = tp_num / (total_gts)
        print("precision_value: {0}".format(precision_value))
        print("recall_value: {0}".format(recall_value))
        print("total_gt_number: {0}".format(total_gts))

        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        # print("FP: {0}".format(fp.size))
        tp = np.cumsum(tp)
        # print("TP: {0}".format(tp.size))
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap

        # print(recall, precision)

    return all_ap, precision, recall, fp, tp


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
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
# model.load_state_dict(torch.load('/home/cen/PycharmProjects/FCOS-PyTorch-37.2AP/checkpoint60/model_30.pth'))

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
               "./20201208fewcheckpointcrop1024/model_{}.pth".format(epoch + 1))
    if epoch > 30 and epoch%10==0:
        # 进行验证
        # 进行验证
        pass

record_pd.to_csv('./loss/20201208fewcheckpointcrop1024loss/loss.csv',index=0)



