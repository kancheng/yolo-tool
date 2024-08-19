# import os
# import subprocess
# import time
# import datetime

import os
import subprocess
import time
import datetime
import base64
from PIL import Image
import json, yaml, argparse
import shutil

files = []
info_files = []
files_check = []
# input_folder = './yolo_runs_202X.../segment/predict'
input_folder = './yolo_runs_XXXXX/segment/predict'
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
yolo_datasets_yaml_path = '/mnt/.../dataset.yaml'
predict_img_path = '/mnt/.../...'
# yolo_datasets_yaml_path = '/mnt/.../dataset.yaml'
# predict_img_path = '/mnt/.../...'

# print(equal_lists)

# for filename in equal_lists:
#     new_files.append(input_folder_labels + "/" + filename +".txt")
#     new_images.append(predict_img_path + "/" + filename +".png")
# print("INFO. new_files : ", new_files)
# print("INFO. new_images : ", new_images)

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

ypredict2labelme(data = yolo_datasets_yaml_path, ptxt = input_folder_labels, ppath = predict_img_path, key_list = equal_lists, out=input_folder)
