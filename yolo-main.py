import os
import time
from ultralytics import YOLO
import argparse 
# Args
# EX: python3 yolo-main.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_datasets_yaml_path', help='input annotated directory')
parser.add_argument('--predict_datasets_folder', help='predict folder')
parser.add_argument('--epochs', default=4000,  help='epochs')
parser.add_argument('--batch', default=2,  help='batch')
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

# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
temtargetpath = './yolo_runs_'+t
command = "yolo settings runs_dir='"+ temtargetpath +"'" 
os.system(command)

files = []
info_files = []
# input_folder = args.input_dir
for filename in os.listdir(predict_datasets_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        info_files.append(predict_datasets_folder + filename)
        files.append(filename)
info_log_files = "INFO. Files : " + str(files)
info_log_the_file_of_number = "INFO. The File Of Number : " + str(len(files))
print(info_log_files)
print(info_log_the_file_of_number)

# Train the model
## Load a model
model_seg = YOLO("yolov8n-seg.pt")
## EX: results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
results_yseg_model_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir))+"/weights/best.pt"

if not os.path.exists(results_yseg_model_path):
    info_log_model = "INFO. Model training failed : " + results_yseg_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_yseg_model_path
log_file_path = os.getcwd()+"/"+str(results_yseg.save_dir) + "/yolo_training_log.txt"
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model)
log_file.close()
# Predict
## EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_yseg_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)
