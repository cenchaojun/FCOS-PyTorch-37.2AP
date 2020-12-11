#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 上午9:26
# @Author  : cenchaojun
# @File    : eval_voc1024everymodel.py
# @Software: PyCharm
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

        if len(fp) == 0:
            print("什么也没有")
            all_ap ={}
            precision =[0]
            recall = [0]
            fp = [0]
            tp = [0]
            fp_num = 0
            tp_num = 0
            precision_value = 0
            recall_value = 0

        else:
            all_ap = {}
            indices = np.argsort(-scores)
            fp = fp[indices]
            tp = tp[indices]
            # 可以写一个for循环，但是iwo还是想用判断语句，简单点
            if np.all(fp == 0):
                fp_num = 0
            else:
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

    return all_ap, precision, recall, fp, tp,fp_num,tp_num,total_gts, precision_value, recall_value


if __name__ == "__main__":
    model_name = 'testresnetnofreeze'
    aplist = []
    fplist =[]
    tplist = []
    record_pd = pd.DataFrame(columns=['epoch', 'tp_num', 'fp_num', 'precision_value', 'recall_value', 'total_gts','ap'])
    for modelnumber in range(12):


        from model.fcos import FCOSDetector
        from detect import convertSyncBNtoBN
        from dataset.VOC_dataset import VOCDataset

        eval_dataset = VOCDataset(root_dir='/home/cen/PycharmProjects/dataset/20201203dataset/fewdataset/voc2007',
                                  resize_size=[1024, 1024],
                                  split='1024val', use_difficult=False, is_train=False, augment=None)
        print("INFO===>eval dataset has %d imgs" % len(eval_dataset))
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn)

        model = FCOSDetector(mode="inference")
        # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # print("INFO===>success convert BN to SyncBN")
        # model = torch.nn.DataParallel(model)

        model.load_state_dict(torch.load("./secondmodel/model_{0}0.pth".format(modelnumber + 1), map_location=torch.device('cpu')))
        # model=convertSyncBNtoBN(model)
        # print("INFO===>success convert SyncBN to BN")
        model = model.cuda().eval()
        print("===>success loading model")

        gt_boxes = []
        gt_classes = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        num = 0
        for img, boxes, classes in tqdm(eval_loader):
            with torch.no_grad():
                out = model(img.cuda())
                pred_boxes.append(out[2][0].cpu().numpy())
                pred_classes.append(out[1][0].cpu().numpy())
                pred_scores.append(out[0][0].cpu().numpy())
            gt_boxes.append(boxes[0].numpy())
            gt_classes.append(classes[0].numpy())
            num += 1
            print(num, end='\r')

        pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)


        all_AP, precision, recall, fp, tp,fp_num,tp_num,total_gts, precision_value, recall_value = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.1,
                                                       len(eval_dataset.CLASSES_NAME))


        # FP_num = dict(Counter(fp).items())
        # print("FP_num : {0}".format(FP_num))
        # 之前放在了上面，可是我感觉没什么问题呀，只有一类，放在上面和下面没有什么问题吧
        # 有一个注意点没有想到，那就是这个precision和recall的格式是np.array格式，和之前的还不一样
        # print("precision: {}".format(precision))
        # print("recall: {}".format(recall))
        # print("fp {}".format(np.sum(fp)))
        # print("tp{} ".format(np.sum(tp)))
        # print("precision size {}".format(precision.size))
        # print("recall size {}".format(recall.size))
        # print("precision shape {}".format(precision.shape))
        # print("recall shape {}".format(recall.shape))
        # 我把这个precision和recall的值都保存下来，我就不相信，结果还不一样
        # precision_data = pd.DataFrame(precision)
        # recall_data = pd.DataFrame(recall)
        # 保存为了画曲线
        # precision_data.to_csv('./loss/precision/{0}_precision_model_{1}0.csv'.format(model_name,modelnumber+1))
        # recall_data.to_csv('./loss/recall/{0}_precision_model_{1}0.csv'.format(model_name,modelnumber+1))
        # 输出一下precision和recall的均值，看行不行吧
        # print("precision mean {0}".format(np.mean(precision)))
        # print("recall mean {0}".format(np.mean(recall)))
        # 将precison和recall的曲线保存下来，看看什么情况吧
        # 使用的是OrderedDict这个库
        # pr_purve = OrderedDict(zip(recall,precision))
        # print("pr-purve len {0}".format(len(pr_purve)))
        # df = pd.DataFrame.from_dict(pr_purve,orient='index')
        # df.to_csv('precision_recall_purve.csv')
        print("all classes AP=====>\n")
        for key, value in tqdm(all_AP.items()):
            print('ap for {} is {}'.format(eval_dataset.id2name[int(key)], value))
        mAP = 0.
        for class_id, class_mAP in tqdm(all_AP.items()):
            mAP += float(class_mAP)
        mAP /= (len(eval_dataset.CLASSES_NAME) - 1)
        aplist.append(mAP)
        print("mAP=====>%.3f\n" % mAP)
        new_row = {'epoch': (modelnumber+1)*10, 'tp_num':tp_num , 'fp_num':fp_num, 'precision_value':precision_value, 'recall_value':recall_value, 'total_gts':total_gts,'ap':mAP}
        record_pd = record_pd.append(new_row,ignore_index=True)
    record_pd.to_csv('./loss/{0}_crop1024result.csv'.format(model_name))
