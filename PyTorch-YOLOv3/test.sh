python3 test.py \
--weights_path run/rico/yolov3_ckpt_10.pth \
--model_def config/yolov3-rico.cfg \
--data_config config/rico.data \
--class_path data/rico/classes.names \
--dataset rico \
--split test \
--img_size 608 \
--batch_size 32