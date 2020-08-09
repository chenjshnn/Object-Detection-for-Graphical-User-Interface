
import cv2
import sys
import json
import time
import random
import pytesseract as pyt

import xianyu_utils as utils

random.seed(123)

def merge_text(corners, max_word_gad=40):
    def is_text_line(corner_a, corner_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = corner_a
        (col_min_b, row_min_b, col_max_b, row_max_b) = corner_b
        # on the same line
        if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
            # close distance
            if abs(col_min_b - col_max_a) < max_word_gad or abs(col_min_a - col_max_b) < max_word_gad:
                return True
        return False

    def corner_merge_two_corners(corner_a, corner_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = corner_a
        (col_min_b, row_min_b, col_max_b, row_max_b) = corner_b

        col_min = min(col_min_a, col_min_b)
        col_max = max(col_max_a, col_max_b)
        row_min = min(row_min_a, row_min_b)
        row_max = max(row_max_a, row_max_b)
        return col_min, row_min, col_max, row_max

    changed = False
    new_corners = []
    for i in range(len(corners)):
        merged = False
        for j in range(len(new_corners)):
            if is_text_line(corners[i], new_corners[j]):
                new_corners[j] = corner_merge_two_corners(corners[i], new_corners[j])
                merged = True
                changed = True
                break
        if not merged:
            new_corners.append(corners[i])

    if not changed:
        return corners
    else:
        return merge_text(new_corners)


def resize_label(bboxes, org_height, resize_height, bias=0):
    bboxes_new = []
    scale = resize_height/org_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def ocr(img, resize_height=800, output_path=None, show=False):
    start = time.clock()
    data = pyt.image_to_data(img)

    bboxes = []
    # level|page_num|block_num|par_num|line_num|word_num|left|top|width|height|conf|text
    for d in data.split('\n')[1:]:
        d = d.split()
        if len(d) < 11:
            continue
        conf = d[10]
        if int(conf) != -1:
            bboxes.append([int(d[6]), int(d[7]), int(d[6]) + int(d[8]), int(d[7]) + int(d[9])])
    bboxes = merge_text(bboxes)
    bboxes = resize_label(bboxes, img.shape[0], resize_height)
    resize_img = utils.resize_by_height(img, resize_height)
    utils.draw_bounding_box(resize_img, bboxes, name='ocr', color=(255,6,6), show=show)
    if output_path is not None:
        utils.save_corners_json(output_path + '.json', bboxes, 'Text')
    # print('OCR [%.3fs] %s' % (time.clock() - start, img_path))
    return bboxes
