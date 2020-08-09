import json
import cv2
import numpy as np
from os.path import join as pjoin
import random
import os

import xianyu_utils as utils

random.seed(123)
color_map = {'Text':(255,6,6), 'Non-Text':(6,255,6)}


def incorporate(img, bbox_compos, bbox_text, show=False):

    def merge_two_corners(corner_a, corner_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = corner_a
        (col_min_b, row_min_b, col_max_b, row_max_b) = corner_b

        col_min = min(col_min_a, col_min_b)
        col_max = max(col_max_a, col_max_b)
        row_min = min(row_min_a, row_min_b)
        row_max = max(row_max_a, row_max_b)
        return [col_min, row_min, col_max, row_max]

    corners_compo_refine = []
    compos_class_refine = []

    mark_text = np.full(len(bbox_text), False)
    for a in bbox_compos:
        broad = utils.draw_bounding_box(img, [a])
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        if area_a < 10:
            continue
        new_corner = None
        text_area = 0
        remain = True
        for i in range(len(bbox_text)):
            b = bbox_text[i]
            if (b[2] - b[0]) / img.shape[1] > 0.9 and\
                    (b[3] - b[1]) / img.shape[0] > 0.1:
                continue

            area_b = (b[2] - b[0]) * (b[3] - b[1])
            # get the intersected area
            col_min_s = max(a[0], b[0])
            row_min_s = max(a[1], b[1])
            col_max_s = min(a[2], b[2])
            row_max_s = min(a[3], b[3])
            w = np.maximum(0, col_max_s - col_min_s)
            h = np.maximum(0, row_max_s - row_min_s)
            inter = w * h
            if inter == 0:
                continue

            # calculate IoU
            ioa = inter / area_a
            iob = inter / area_b
            iou = inter / (area_a + area_b - inter)

            # print('ioa:%.3f, iob:%.3f, iou:%.3f' %(ioa, iob, iou))
            # utils.draw_bounding_box(broad, [b], color=(255,0,0), line=2, show=True)

            # text area
            if iou > 0.6:
                # new_corner = merge_two_corners(a, b)
                new_corner = b
                mark_text[i] = True
                break
            if ioa > 0.55:
                remain = False
                break
            # if iob == 1:
            #     text_area += inter

        if new_corner is not None:
            corners_compo_refine.append(new_corner)
            compos_class_refine.append('Text')
        elif text_area / area_a > 0.5:
            corners_compo_refine.append(a)
            compos_class_refine.append('Text')
        elif not remain:
            continue
        else:
            corners_compo_refine.append(a)
            compos_class_refine.append('Compo')

    for i in range(len(bbox_text)):
        if not mark_text[i]:
            corners_compo_refine.append(bbox_text[i])
            compos_class_refine.append('Text')

    if show:
        board = utils.draw_bounding_box_class(img, corners_compo_refine, compos_class_refine)
        cv2.imshow('merge', board)
        cv2.waitKey()

    return corners_compo_refine, compos_class_refine