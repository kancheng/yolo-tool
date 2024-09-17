import os
import time
from ultralytics import YOLO
import argparse 

# arrange an instance segmentation model for test
from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

import torch

# from function.reportf import report_function
from function.sahi_dashboardsf import report_function_d



cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA available.")
    # "cpu" or 'cuda:0'
    nkey = 'cuda:0'
else:
    print("CUDA not available.")
    # "cpu" or 'cuda:0'
    nkey = 'cpu'

# import base64
# from PIL import Image
# import json, yaml, argparse
# import shutil

# from function.reportf import report_function
# from function.dashboardsf import report_function_d

# Args
# EX: python3 yolo-main.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_datasets_yaml_path', help='input annotated directory')
parser.add_argument('--predict_datasets_folder', help='predict folder')
parser.add_argument('--name', default='dl',  help='project name')
parser.add_argument('--epochs', default=4000,  help='epochs')
parser.add_argument('--batch', default=2,  help='batch')
parser.add_argument('--models', default='yolov8n-seg',  help='models name')
args = parser.parse_args()

# Settings Path.
# input_datasets_yaml_path = '/mnt/ ... /dataset.yaml'
input_datasets_yaml_path = args.input_datasets_yaml_path
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder
# epochs
epochs_num = int(args.epochs)
# batch
batch_num = int(args.batch)
# name 
project_name = args.name
# models name
models_name = args.models
models_key = ""
info_log_model_type = ""
if models_name == 'yolov8n-seg' :
    models_key = './models/' + 'yolov8n-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8l-seg' :
    models_key = './models/' + 'yolov8l-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8m-seg' :
    models_key = './models/' + 'yolov8m-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8s-seg' :
    models_key = './models/' + 'yolov8s-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8x-seg' :
    models_key = './models/' + 'yolov8x-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
else :
    models_key = './models/' + 'yolov8n-seg.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)

# print(models_key)
# print(models_name)
# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
# temtargetpath = './yolo_runs_'+t
p = os.getcwd()
temtargetpath = p + '/yolo_runs_sahi_'+ models_name +'_'+ project_name +'_'+ t
command = "yolo settings runs_dir='"+ temtargetpath +"'" 
os.system(command)
# print(temtargetpath)

files = []
info_files = []
# input_folder = args.input_dir
for filename in os.listdir(predict_datasets_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        info_files.append(predict_datasets_folder + "/"+ filename)
        files.append(filename)
info_log_files = "INFO. Files : " + str(files)
info_log_the_file_of_number = "INFO. The File Of Number : " + str(len(files))
print(info_log_files)
print(info_log_the_file_of_number)

# Train the model
## Load a model
# model_seg = YOLO("yolov8n-seg.pt")
model_seg = YOLO(models_key)

## EX: results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
results_yseg_model_path = str(results_yseg.save_dir) + "/weights/best.pt"
if not os.path.exists(results_yseg_model_path):
    info_log_model = "INFO. Model training failed : " + results_yseg_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_yseg_model_path
# log_file_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file_path = os.path.dirname(str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model + '\n' + info_log_model_type + '\n' + 'INFO. Work : SAHI Seg.')
log_file.close()

# SAHI Predict.

model_config_path = str(results_yseg.save_dir) + "/args.yaml" # agnostic_nms=True in the .yaml file
predict(
model_type="yolov8",
model_path=results_yseg_model_path,
model_config_path=model_config_path,
model_device=nkey, # "cpu" or 'cuda:0'
model_confidence_threshold=0.6,
postprocess_class_agnostic=True,
source=predict_datasets_folder,
slice_height=640,
slice_width=640,
overlap_height_ratio=0.2,
overlap_width_ratio=0.2,
visual_bbox_thickness=1,
visual_text_size=0.5,
visual_text_thickness=1,
export_pickle=True,
project= os.path.dirname(str(results_yseg.save_dir)) + "/predict"
)

detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=results_yseg_model_path,
    confidence_threshold=0.3,
    device=nkey, # or 'cuda:0'
)


# Predict SAHI sliced .
# EX :
# result = get_sliced_prediction(
#     "demo_data/small-vehicles1.jpeg",
#     detection_model_seg,
#     slice_height = 256,
#     slice_width = 256,
#     overlap_height_ratio = 0.2,
#     overlap_width_ratio = 0.2
# )

sahi_path = os.path.dirname(str(results_yseg.save_dir)) + "/sliced_prediction"
# if not os.path.exists(sahi_path):
#     os.makedirs(sahi_path)

for filename in info_files:
    # print(filename)
    files_key = filename.split(".")[0]
    sahi_files_key = os.path.basename(filename)
    sahi_filenames_key = sahi_files_key.split(".")[0]

    result = get_sliced_prediction(
        filename,
        detection_model_seg,
        slice_height = 256,
        slice_width = 256,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    result.export_visuals(export_dir=(sahi_path + "/"), file_name=sahi_filenames_key)
    # print(result.object_prediction_list)
    # print(dir(result.object_prediction_list))
    # print(len(result.object_prediction_list))
    # print(result.object_prediction_list[0])
    # print(dir(result.object_prediction_list[0]))
    # print(len(result.object_prediction_list[0]))



# ymal_path = '/mnt/... /dataset.yaml'
# ymal_path = input_datasets_yaml_path

# original_image = '/mnt/ ... /yolov8-datasets-predict-name'
original_image = predict_datasets_folder

# predict_folder = './yolo_runs_.../segment/predict'
predict_folder = os.path.dirname(str(results_yseg.save_dir)) + "/sliced_prediction"

# train_folder = './yolo_runs_.../segment/train'
train_folder = os.path.dirname(str(results_yseg.save_dir)) + "/train"

# html_path = "./yolo_runs_.../segment/index-report.html"
html_path = os.path.dirname(str(results_yseg.save_dir)) + "/dashboards_sahi.html"

# pout = "./yolo_runs_.../yolo2images"
pout = os.path.dirname(str(results_yseg.save_dir)) + "/yolo2images"

report_function_d(input_datasets_yaml_path, original_image, predict_folder, train_folder, html_path, pout)

