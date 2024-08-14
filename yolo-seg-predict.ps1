# USE *.ps1
# Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

python yolo-seg-predict.py

# python yolo-seg-predict.py --input_dir="./inputs/" --model_path="/mnt/../../yolov8/runs/segment/train/weights/best.pt"