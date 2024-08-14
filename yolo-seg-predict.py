import os
import subprocess
import time
import datetime

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', help='input annotated directory')
parser.add_argument('--model_path', help='model path')
args = parser.parse_args()

# args.input_dir
# input_folder =  '/mnt/.../'
input_folder =  args.input_dir
# args.model_path
# model_path = "./.../"
model_path = args.model_path

def get_platform():
    import platform
    sys_platform = platform.platform().lower()
    if "windows" in sys_platform:
        print("Windows")
    elif "macos" in sys_platform:
        print("Mac os")
    elif "linux" in sys_platform:
        print("Linux")
    else:
        print("Others")

files = []
info_files = []
input_folder = args.input_dir
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        info_files.append(input_folder+filename)
        files.append(filename) 
print("INFO. Files : ", files)


t = time.strftime("%Y%m%d%H%M%S", time.localtime())
temtargetpath = './yolo_predict_runs_'+t
command = "cd yolov8 && yolo settings runs_dir='"+ temtargetpath +"'" 
os.system(command)

# EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

for filename in info_files:
    print( "INFO. PREPARE THE COMMAND TO BE EXECUTED : yolo segment predict model='" + model_path + "' source='" + filename + "' save_txt=True")
    temcmd = "yolo segment predict model='"+ model_path + "' source='" + filename + "' save_txt=True"
    os.system(temcmd)



