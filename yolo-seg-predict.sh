#!/bin/bash

# YOLOV8
# yolo settings runs_dir='/mnt/div/path'
# bash yolo-seg-predict.sh /mnt/div/pic-path /mnt/div/yolo/runs/segment/trainXX/weights/best.pt

function read_dir(){

for file in `ls $1` # Note that these are two backticks here, indicating running system commands
    do
        if [ -d $1"/"$file ] # Note that there must be spaces between here, otherwise an error will be reported.
        then
        read_dir $1"/"$file
        else
        echo "INFO. FILE PATH: " $1"/"$file # Just process the file here
        echo "INFO. PREPARE THE COMMAND TO BE EXECUTED : yolo segment predict model='"$2"' source='"$1"/"$file"' save_txt=True"
        yolo segment predict model=$2 source=$1"/"$file save_txt=True
        # yolo
        fi
    done
}
 
read_dir $1 $2
