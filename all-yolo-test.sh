# WSL YOLO ENV.

echo '# ---------- yolov11l-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov11l-seg" --epochs 100

echo '# ---------- yolov11m-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov11m-seg" --epochs 100

echo '# ---------- yolov11n-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov11n-seg" --epochs 100

echo '# ---------- yolov11s-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov11s-seg" --epochs 100

echo '# ---------- yolov11x-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov11x-seg" --epochs 100

echo '# ---------- yolov8n-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov8n-seg" --epochs 100

echo '# ---------- yolov8l-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov8l-seg" --epochs 100

echo '# ---------- yolov8m-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov8m-seg" --epochs 100

echo '# ---------- yolov8s-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov8s-seg" --epochs 100

echo '# ---------- yolov8x-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov8x-seg" --epochs 100

echo '# ---------- yolov9c-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov9c-seg" --epochs 100

echo '# ---------- yolov9e-seg ---------- #'

python3 yolo-main.py --input_datasets_yaml_path="/mnt/e/yolo/yolo-datasets/dataset.yaml" --predict_datasets_folder="/mnt/e/yolo/yolo-datasets-predict/datasets-predict-name" --models="yolov9e-seg" --epochs 100