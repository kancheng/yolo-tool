import os
import time
from ultralytics import YOLO
import argparse 
# Args
# EX: python3 yolo-main.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_datasets_yaml_path', help='input annotated directory')
parser.add_argument('--predict_datasets_folder', help='predict folder')
args = parser.parse_args()

# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
temtargetpath = './yolo_runs_'+t
command = "yolo settings runs_dir='"+ temtargetpath +"'" 
print("INFO. Log : ", command)
os.system(command)

# Settings Path.
# input_datasets_yaml_path = '/mnt/ ... /dataset.yaml'
input_datasets_yaml_path = args.input_datasets_yaml_path
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder
files = []
info_files = []
# input_folder = args.input_dir
for filename in os.listdir(predict_datasets_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        info_files.append(predict_datasets_folder + filename)
        files.append(filename) 
print("INFO. Files : ", files)
print("INFO. The File Of Number : ", len(files))

# Train the model
## Load a model
model_seg = YOLO("yolov8n-seg.pt")
## EX: results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=4000, imgsz=640, batch=2)
results_yseg_model_path = os.getcwd()+"/"+str(results_yseg.save_dir)+"/weights/best.pt"


# Predict
## EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_yseg_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)
