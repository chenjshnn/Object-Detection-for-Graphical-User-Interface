python3 detect.py  \
--weights_path checkpoints/yolov3_ckpt_42.pth \
--model_def config/yolov3-custom.cfg \
--class_path data/custom/classes.names \
--image_folder data/custom/images


--data_config config/custom.data \