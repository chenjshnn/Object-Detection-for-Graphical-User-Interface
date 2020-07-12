from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import json, tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from pycocotools.coco import COCO


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, dataset_name, split = "val", epoch=''):

    gt_json = "data/{}/annotations/instances_{}.json".format(dataset_name, split)
    gt_COCO = COCO(gt_json)


    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    results = []
    all_labels = set()
    max_width = 0
    max_height = 0

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    for batch_i, (img_paths, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # if batch_i == 2:
        #     break

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(input_imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)


        imgs.extend(img_paths)
        img_detections.extend(outputs)

        # print(len(outputs), len(outputs[0]), len(outputs[0][0]))
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    results = []
    for img_i, (path, detections) in enumerate(tqdm.tqdm(zip(imgs, img_detections))):

        imgid = int(os.path.basename(path).replace(".jpg", ""))

        # print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))


        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = list(map(lambda x:x.item(), [x1, y1, x2, y2, conf, cls_conf, cls_pred]))
                x1 = x1 if x1 >= 0 else 0
                y1 = y1 if y1 >= 0 else 0
                box_w = x2 - x1 
                box_h = y2 - y1
                # print(x1,x2,box_w, box_h)
                item = {"image_id":imgid,
                        "category_id": int(cls_pred),
                        "score": conf,
                        "bbox":[round(x1), round(y1), round(box_w), round(box_h)],
                        }

                results.append(item)  

    # targets (imgidx, label, x1, y1, x2, y2)
    # print(targets)

    # outputs (x1, y1, x2, y2, score, label)
    
    print(all_labels)
    print(max_width, max_height)

    results_root = "results/output/{}_{}/".format(dataset_name, split)
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    with open(os.path.join(results_root, "{}_{}_results{}.json".format(dataset_name, split, epoch)), "w") as f:
        json.dump(results, f)
    # Concatenate sample statistics
    # print("error:", sample_metrics)
    if len(sample_metrics)> 0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    else:
        return 0,0,0,0,0

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--dataset", type=str, default="dataset", help="dataset")
    parser.add_argument("--split", type=str, default="valid", help="dataset")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid" if opt.split == "val" else opt.split]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        dataset_name = opt.dataset,
        split = opt.split

    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP: {}".format(c, class_names[c], AP[i]))

    print("mAP: {}".format(AP.mean()))
