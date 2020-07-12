# faster rcnn
import sys, json
sys.path.append("../FASTER_RCNN/lib")
from model.roi_layers import nms

import torch
from torch.autograd import Variable
import numpy as np

def xywh2xyxy(box):
	# print(box)
	x,y,w,h= box
	return [x,y,x+w,y+h]

def nms_for_results(result_json, nms_threshold, output_json):
	all_boxes = json.load(open(result_json, "r"))
	print("Before NMS:", len(all_boxes))
	# reformat
	all_data = {}
	for item in all_boxes:
		imgid = item["image_id"]
		if imgid not in all_data:
			all_data[imgid] = []
		all_data[imgid].append(item)

	num_images = len(all_data)

	after_nms = []
	for i, imgid in enumerate(all_data.keys()): #
		all_items = all_data[imgid]

		all_items.sort(key=lambda x:x["score"], reverse=True)
		pred_boxes = list(map(lambda x:xywh2xyxy(x["bbox"]), all_items))
		cls_scores = list(map(lambda x:x["score"], all_items))

		pred_boxes = Variable(torch.Tensor(pred_boxes))
		cls_scores = Variable(torch.Tensor(cls_scores))

		cls_dets = torch.cat((pred_boxes, cls_scores.unsqueeze(1)), 1)

		keep = nms(pred_boxes, cls_scores, nms_threshold)
		keep = keep.view(-1).long().cpu()

		keep_items = list(map(lambda x:all_items[x], keep))

		after_nms.extend(keep_items)

	print("After NMS:", len(after_nms))
	with open(output_json, "w") as f:
		json.dump(after_nms, f)

