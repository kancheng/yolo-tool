import os
import time
from ultralytics import YOLO
import argparse 

import base64
from PIL import Image
import json, yaml, argparse
import shutil

from function.reportf import report_function

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
# models_key = './models/' + 
if models_name == 'yolov8n-seg' :
    models_key = './models/' + 'yolov8n-seg.pt'
elif models_name == 'yolov8l-seg' :
    models_key = './models/' + 'yolov8l-seg.pt'
elif models_name == 'yolov8l' :
    models_key = './models/' + 'yolov8l.pt'
elif models_name == 'yolov8m-seg' :
    models_key = './models/' + 'yolov8m-seg.pt'
elif models_name == 'yolov8m' :
    models_key = './models/' + 'yolov8m.pt'
elif models_name == 'yolov8n' :
    models_key = './models/' + 'yolov8n.pt'
elif models_name == 'yolov8s-seg' :
    models_key = './models/' + 'yolov8s-seg.pt'
elif models_name == 'yolov8s' :
    models_key = './models/' + 'yolov8s.pt'
elif models_name == 'yolov8x-seg' :
    models_key = './models/' + 'yolov8x-seg.pt'
elif models_name == 'yolov8x' :
    models_key = './models/' + 'yolov8x.pt'
elif models_name == 'yolov10b' :
    models_key = './models/' + 'yolov10b.pt'
elif models_name == 'yolov10l' :
    models_key = './models/' + 'yolov10l.pt'
elif models_name == 'yolov10m' :
    models_key = './models/' + 'yolov10m.pt'
elif models_name == 'yolov10n' :
    models_key = './models/' + 'yolov10n.pt'
elif models_name == 'yolov10s' :
    models_key = './models/' + 'yolov10s.pt'
elif models_name == 'yolov10x' :
    models_key = './models/' + 'yolov10x.pt'
elif models_name == 'yolov9c-seg' :
    models_key = './models/' + 'yolov9c-seg.pt'
elif models_name == 'yolov9c' :
    models_key = './models/' + 'yolov9c.pt'
elif models_name == 'yolov9e-seg' :
    models_key = './models/' + 'yolov9e-seg.pt'
elif models_name == 'yolov9e' :
    models_key = './models/' + 'yolov9e.pt'
elif models_name == 'yolov9m' :
    models_key = './models/' + 'yolov9m.pt'
elif models_name == 'yolov9s' :
    models_key = './models/' + 'yolov9s.pt'
elif models_name == 'yolov9t' :
    models_key = './models/' + 'yolov9t.pt'

# print(models_key)
# print(models_name)
# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
# temtargetpath = './yolo_runs_'+t
p = os.getcwd()
temtargetpath = p + '/yolo_runs_'+ models_name +'_'+ project_name +'_'+ t
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
results_yseg_model_path = os.getcwd()+"/"+str(results_yseg.save_dir)+"/weights/best.pt"
if not os.path.exists(results_yseg_model_path):
    info_log_model = "INFO. Model training failed : " + results_yseg_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_yseg_model_path
log_file_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model)
log_file.close()
# Predict
## EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_yseg_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)

# YOLO Predict Label to Labelme JSON

files = []
info_files = []
files_check = []
input_folder = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) +'/predict'
input_folder_labels = input_folder + '/labels'
for filename in os.listdir(input_folder_labels):
    if filename.endswith((".txt")):
        info_files.append(input_folder_labels + "/" + filename)
        files.append(filename) 
        for con in files:
            files_check.append(con.split(".")[0])
print("INFO. Files : ", files)
print("INFO. The File Of Number : ", len(files))
# print("INFO. TXT Path : ", info_files)

images = []
info_images = []
images_check = []
for filename in os.listdir(input_folder):
    if filename.endswith((".png")):
        info_images.append(input_folder + "/" + filename)
        images.append(filename)
        for con in images:
            images_check.append(con.split(".")[0])
# print("INFO. Images: ", images)
print("INFO. The Images Of Number : ", len(images))
# print("INFO. Images Path : ", info_images)
equal_lists = [x for x in files_check if x in images_check]

# To labelme Dir.
image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

def get_shapes(txt_path, width, height, class_labels):
    shapes = open(txt_path).read().split('\n')
    result = []
    for shape in shapes:
        if not shape:
            continue
        values = shape.split()

        class_id = values[0]
        r_shape = dict()
        r_shape["label"] = class_labels[int(class_id)]

        values = [float(value) for value in values[1:]]
        points = []
        for i in range(len(values)//2):
            points.append([values[2*i]*width, values[2*i+1]*height])
        r_shape['points'] = points

        r_shape.update({ "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        })
        result.append(r_shape)
    return result

def tobase64(file_path):
    with open(file_path, "rb") as image_file:
        data = base64.b64encode(image_file.read())
        return data.decode()

def img_filename_to_ext(img_filename, ext='txt'):
    for img_ext in image_extensions:
        if img_filename.lower().endswith(img_ext):
            return img_filename[:-len(img_ext)] + ext

def is_image_file(file_path):
    file_path = file_path.lower()
    for ext in image_extensions:
        if file_path.endswith(ext):
            return True
    return False

def yolo2labelme_single(txt_path, img_path, class_labels, out_dir):
    img = Image.open(img_path)
    result = {"version": "5.2.1", "flags": {}}
    result['shapes'] = get_shapes(txt_path, img.width, img.height, class_labels)
    result["imagePath"] = img_path
    result["imageData"] = tobase64(img_path)
    result["imageHeight"] = img.height
    result["imageWidth"] = img.width

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    img_filename = os.path.basename(img_path)
    json_path = img_filename_to_ext(img_filename,'.json')
    json_path = os.path.join(out_dir,json_path)
    with open(json_path,'w') as f:
        f.write(json.dumps(result))
    shutil.copyfile(img_path, os.path.join(out_dir, img_filename) )

new_files = []
new_images = []
# yolo_datasets_yaml_path = input_datasets_yaml_path
# predict_img_path = predict_datasets_folder
# '/mnt/w/w/w/e/' -> '/mnt/w/w/w/e'
# predict_img_path = os.path.split(predict_datasets_folder)[0]

def ypredict2labelme(data, ptxt, ppath, key_list, out, skip=False,):
    yaml_path = os.path.join(data)
    with open(yaml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        class_labels = data_loaded['names']
    print(class_labels)
    out = out + '/labelmeDataset'
    new_files = []
    new_images = []
    for filename in key_list:
        tmtxts = ptxt + "/" + filename +".txt"
        tmimages = ppath + "/" + filename +".png"
        new_files.append(ptxt + "/" + filename +".txt")
        new_images.append(ppath + "/" + filename +".png")
        # print("INFO. new_files : ", new_files)
        # print("INFO. new_images : ", new_images)
        yolo2labelme_single(tmtxts, tmimages, class_labels, out)
    print("INFO. new_files : ", len(new_files))
    print("INFO. new_images : ", len(new_images))

ypredict2labelme(data = input_datasets_yaml_path, ptxt = input_folder_labels, ppath = predict_datasets_folder , key_list = equal_lists, out=input_folder)

# ymal_path = '/mnt/... /dataset.yaml'
# ymal_path = input_datasets_yaml_path

# original_image = '/mnt/ ... /yolov8-datasets-predict-name'
original_image = predict_datasets_folder

# predict_folder = './yolo_runs_.../segment/predict'
predict_folder = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/predict"

# train_folder = './yolo_runs_.../segment/train'
train_folder = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/train"

# html_path = "./yolo_runs_.../segment/index-report.html"
html_path = os.path.dirname(os.getcwd() + "/" + str(results_yseg.save_dir)) + "/report.html"

# pout = "./yolo_runs_.../yolo2images"
pout = os.path.dirname(os.getcwd() + "/" + str(results_yseg.save_dir)) + "/yolo2images"

report_function(input_datasets_yaml_path, original_image, predict_folder, train_folder, html_path, pout)
