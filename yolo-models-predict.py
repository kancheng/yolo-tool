import os
import time
from ultralytics import YOLO
import argparse 

# Args
# EX: python3 yolo-main.py --predict_datasets_folder="/mnt/.../ " --models="" 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--predict_datasets_folder', help='predict folder')
parser.add_argument('--name', default='dl',  help='project name')
parser.add_argument('--models', default='yolov8n-seg',  help='models name')
args = parser.parse_args()

# Settings Path.
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder
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
print(models_name)

# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
p = os.getcwd()
temtargetpath = p + '/yolo_models_predict_' + t
command = "yolo settings runs_dir='"+ temtargetpath +"'" 
os.system(command)


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

model_predict = YOLO(models_name, task='segment')

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)

