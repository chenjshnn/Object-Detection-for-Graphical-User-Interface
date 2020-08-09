#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import imagesize
from tqdm import tqdm
import cv2

from nms import nms_for_results


from config import system_configs
from nnet.py_factory import NetworkFactory
from db.rico_single import RICO_DEMO
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CenterNet")
    parser.add_argument("--cfg_file", help="config file", type=str, \
                        choices=["CenterNet-52-rico", "CenterNet-52-rico2k", "CenterNet-52-rico10k", \
                                 "CenterNet-52-ricotext"])
    parser.add_argument("--test_folder", dest="test_folder",
                        help="which test_folder to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true", default = False)

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def visualize(imgname2id, nms_json, thresh=0.28):
    def get_score(item):
        return item["score"]

    predicted_box = json.load(open(nms_json, "r"))
    # print(predicted_box)

    collect_all_data = {}
    for item in tqdm(predicted_box):
        imgid = item["image_id"]
        score = item["score"]

        if imgid not in collect_all_data:
            collect_all_data[imgid] = []
        # print(item)
        if score > 0.28:
            collect_all_data[imgid].append(item)
            print(item)

    for imgpath in imgname2id:
        im = cv2.imread(imgpath)
        imgid = imgname2id[imgpath]

        if imgid not in collect_all_data:
            continue
        detections = collect_all_data[imgid]
        detections.sort(key=get_score, reverse=True)
        print("draw", detections[:3])
        for item in detections[:70]:

            # score = item["score"]
            bbox = item["bbox"]
            class_name = item["category_id"]
            # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

            bbox = tuple(int(np.round(x)) for x in bbox[:4])

            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            score = item["score"]

            cv2.rectangle(im, bbox[0:2], (x2,y2), (0, 255, 0), 2)


        cv2.imwrite(".".join(imgpath.split(".")[:-1])+"-detected.png", im)

def test(db, split, testiter, num_classes, result_dir, imgname2id, debug=False, suffix=None): 

    print("resu", result_dir)
    make_dirs([result_dir])

    print("building neural network...")
    nnet = NetworkFactory(db, num_classes)
    print("loading parameters...")
    nnet.load_params("")

    # test_file = "test.{}".format(db.data)

    test_file = "test.coco"
    testing = importlib.import_module(test_file).testing

    nnet.cuda()
    nnet.eval_mode()
    testing(db, nnet, result_dir, debug=debug)
    result_json  = os.path.join(result_dir, "results.json")
    nms_json = os.path.join(result_dir, "results-nms.json")
    nms_for_results(result_json, 0.7, nms_json)
    visualize(imgname2id, nms_json, thresh=0.28)

if __name__ == "__main__":
    args = parse_args()
    split = "tmp"
    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = "-".join(args.cfg_file.split("-")[:2]) # get the model name
    system_configs.update_config(configs["system"])
    system_configs.update_config({"categories":configs["db"]["categories"], "current_split":split})

    

    ### load testing data
    all_test_imgs = os.listdir(args.test_folder)
    tmp_json_dict = {"annotation":[],
                     "images":[],
                     "categories":[{'id': 0, 'supercategory': 'Button', 'name': 'Button'}, {'id': 1, 'supercategory': 'CheckBox', 'name': 'CheckBox'}, {'id': 2, 'supercategory': 'Chronometer', 'name': 'Chronometer'}, {'id': 3, 'supercategory': 'EditText', 'name': 'EditText'}, {'id': 4, 'supercategory': 'ImageButton', 'name': 'ImageButton'}, {'id': 5, 'supercategory': 'ImageView', 'name': 'ImageView'}, {'id': 6, 'supercategory': 'ProgressBar', 'name': 'ProgressBar'}, {'id': 7, 'supercategory': 'RadioButton', 'name': 'RadioButton'}, {'id': 8, 'supercategory': 'RatingBar', 'name': 'RatingBar'}, {'id': 9, 'supercategory': 'SeekBar', 'name': 'SeekBar'}, {'id': 10, 'supercategory': 'Spinner', 'name': 'Spinner'}, {'id': 11, 'supercategory': 'Switch', 'name': 'Switch'}, {'id': 12, 'supercategory': 'ToggleButton', 'name': 'ToggleButton'}, {'id': 13, 'supercategory': 'VideoView', 'name': 'VideoView'}]
                    }


    curr_id = 0
    imgname2id = {}
    for imgpath in all_test_imgs:
        if imgpath.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
            if "detected" in imgpath:
                continue
            full_path = os.path.join(args.test_folder, imgpath)
            w, h = imagesize.get(full_path)
            curr_item = {"id":curr_id,
                         "file_name":full_path,
                         "height":h, 
                         "width":w}
            imgname2id[full_path] = curr_id
            tmp_json_dict["images"].append(curr_item)
            curr_id += 1

    with open(os.path.join(args.test_folder, "instances_tmp.json"), "w") as f:
        json.dump(tmp_json_dict, f)

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    testing_db = RICO_DEMO(configs["db"], split, args.test_folder)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    test(testing_db, split, "", configs["db"]["categories"], args.test_folder, imgname2id, args.debug, args.suffix)
