import os
from PIL import Image
import time
import base64
import cv2
import imutils
import shutil
import numpy as np
import argparse

# Args
# EX: python3 report.py --ypath="/mnt/.../dataset.yaml" --original="/mnt/.../yolov8-datasets-predict" --ywork="./yolo_runs_.../segment" --output="./yolo_runs_.../segment/predict/index.html" --yimg="./yolo_runs_.../yolo2images"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ypath')
parser.add_argument('--original')
parser.add_argument('--ywork')
parser.add_argument('--yimg', default="./yolo2images")
parser.add_argument('--output', default="./index.html")
args = parser.parse_args()

def read_txt_labels(txt_file):
    """
    Read labels from txt annotation file
    :param txt_file: txt annotation file path
    :return: tag list
    """
    with open(txt_file, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split(" ")
            class_id = int(label_data[0])
            # Parsing bounding box coordinates
            coordinates = [float(x) for x in label_data[1:]]
            labels.append([class_id, coordinates])
    return labels

def draw_labels(image, labels):
    """
    Draw segmentation region outlines on the image
    :param image: image
    :param labels: list of labels
    """
    for label in labels:
        class_id, coordinates = label
        # Convert coordinates to integers and reshape into polygons
        points = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in zip(coordinates[::2], coordinates[1::2])]
        # Draw outlines using polygons
        cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2) # Red indicates the segmentation area outline

def yolo2imagesbase64( pimg, ptxt):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    # Reading an Image
    # image = cv2.imread("./test/coco128.jpg")
    image = cv2.imread(pimg)
    # Read txt annotation file
    # txt_file = "./test/coco128.txt"
    height, width, _  = image.shape
    txt_file = ptxt
    labels = read_txt_labels(txt_file)
    # Draw segmentation area
    draw_labels(image, labels)
    # Get the window size
    window_size = (width//2, height//2) # You can resize the window as needed
    # Resize an image
    image = cv2.resize(image, window_size)
    # Create a black image the same size as the window
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    # Place the image in the center of the black background
    image_x = int((window_size[0] - image.shape[1]) / 2)
    image_y = int((window_size[1] - image.shape[0]) / 2)
    background[image_y:image_y + image.shape[0], image_x:image_x + image.shape[1]] = image
    # Using cv2.imwrite() method
    # Saving the image
    return image

def compress_images(input_dir, output_dir, quality=85):
    """
    Compress all images in the input directory and save them to the output directory.
    Parameters:
    input_dir (str): The path to the input directory.
    output_dir (str): Path to the output directory.
    quality (int): The quality of compression, ranging from 0 to 100. The higher the value, the better the quality.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Open picture
            img = Image.open(input_path)
            # The length and width are reduced to 25%
            width, height = img.size
            new_size = (width//4, height//4)
            resized_image = img.resize(new_size)
            # Save
            resized_image.save(output_path, quality=quality)
            
            print(f'{filename} compressed and saved to {output_dir}')

def yolo2images( pimg, ptxt, out):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    # Reading an Image
    # image = cv2.imread("./test/coco128.jpg")
    image = cv2.imread(pimg)
    # Read txt annotation file
    # txt_file = "./test/coco128.txt"
    height, width, _  = image.shape
    txt_file = ptxt
    labels = read_txt_labels(txt_file)
    # Draw segmentation area
    draw_labels(image, labels)
    # Get the window size
    # window_size = (width//2, height//2) # You can resize the window as needed
    window_size = (width, height) # You can resize the window as needed
    # Resize an image
    image = cv2.resize(image, window_size)
    # Create a black image the same size as the window
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    # Place the image in the center of the black background
    image_x = int((window_size[0] - image.shape[1]) / 2)
    image_y = int((window_size[1] - image.shape[0]) / 2)
    background[image_y:image_y + image.shape[0], image_x:image_x + image.shape[1]] = image

    # Filename
    # filename = 'savedImage.jpg'
    filename = out

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, image)

def report_function_d(ymal_path, original_image, predict_folder, train_folder, html_path, pout):
    predict_folder_labels = predict_folder + "/labels"
    up_ymal_path = os.path.dirname(ymal_path)
    ptxt = up_ymal_path + "/labels"
    pimg = up_ymal_path + "/images"

    rawdir = pout+"/raw"
    resdir = pout+"/res"

    if not os.path.exists(pout):
        os.makedirs(rawdir)
        os.makedirs(resdir)
    '''
    Read txt annotation files and original images
    '''
    ytxt = []
    yimg = []

    up_ymal_path_img_t = pimg + "/train"
    up_ymal_path_img_v = pimg + "/val"
    for filename in os.listdir(up_ymal_path_img_t):
        if filename.endswith((".png")):
            yimg.append(up_ymal_path_img_t + "/" + filename)
            
    for filename in os.listdir(up_ymal_path_img_v):
        if filename.endswith((".png")):
            yimg.append(up_ymal_path_img_v + "/" + filename)

    up_ymal_path_txt_t = ptxt + "/train"
    up_ymal_path_txt_v = ptxt + "/val"
    for filename in os.listdir(up_ymal_path_txt_t):
        if filename.endswith((".txt")):
            ytxt.append(up_ymal_path_txt_t + "/" + filename)

    for filename in os.listdir(up_ymal_path_txt_v):
        if filename.endswith((".txt")):
            ytxt.append(up_ymal_path_txt_v + "/" + filename)
    temd = []
    for i in range(len(ytxt)):
        shutil.copyfile(ytxt[i],(rawdir +"/"+ os.path.basename(ytxt[i])))
        tname = os.path.basename(ytxt[i])
        temd.append(tname.split(".")[0])

    for i in range(len(yimg)):
        shutil.copyfile(yimg[i],(rawdir +"/"+ os.path.basename(yimg[i])))

    for filename in temd:
        simg = rawdir + "/" + filename + ".png"
        stxt = rawdir + "/" + filename + ".txt"
        sout = resdir + "/" + filename + ".png"
        yolo2images( pimg = simg, ptxt= stxt, out =sout)
    # time
    st = time.strftime("%Y.%m.%d", time.localtime())
    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    cont_head ="<!DOCTYPE html><head><meta charset='utf-8'><meta http-equiv='X-UA-Compatible' content='IE=edge'><meta name='viewport' content='width=device-width, initial-scale=1'><meta name='description' content='Report'><meta name='author' content='koala'><title>Report</title><style>html{font-family: sans-serif; line-height: 1.15; -ms-text-size-adjust: 100%; -webkit-text-size-adjust: 100%; scroll-behavior: smooth;} body{margin: 0;} article, aside, footer, header, nav, section{display: block;} h1{font-size: 2em; margin: 0.67em 0;} figcaption, figure, main{display: block;} figure{margin: 1em 40px;} hr{box-sizing: content-box; height: 0; overflow: visible;} pre{font-family: monospace, monospace; font-size: 1em;} a{background-color: transparent; -webkit-text-decoration-skip: objects;} a:active, a:hover{outline-width: 0;} abbr[title]{border-bottom: none; text-decoration: underline; text-decoration: underline dotted;} b, strong{font-weight: inherit;} b, strong{font-weight: bolder;} code, kbd, samp{font-family: monospace, monospace; font-size: 1em;} dfn{font-style: italic;} mark{background-color: #ff0; color: #000;} small{font-size: 80%;} sub, sup{font-size: 75%; line-height: 0; position: relative; vertical-align: baseline;} sub{bottom: -0.25em;} sup{top: -0.5em;} audio, video{display: inline-block;} audio:not([controls]){display: none; height: 0;} img{border-style: none; border-radius: 3px;} svg:not(:root){overflow: hidden;} button, input, optgroup, select, textarea{font-family: sans-serif; font-size: 100%; line-height: 1.15; margin: 0;} button, input{overflow: visible;} button, select{text-transform: none;} button, html [type='button'], [type='reset'], [type='submit']{-webkit-appearance: button;} button::-moz-focus-inner, [type='button']::-moz-focus-inner, [type='reset']::-moz-focus-inner, [type='submit']::-moz-focus-inner{border-style: none; padding: 0;} button:-moz-focusring, [type='button']:-moz-focusring, [type='reset']:-moz-focusring, [type='submit']:-moz-focusring{outline: 1px dotted ButtonText;} fieldset{border: 1px solid #c0c0c0; margin: 0 2px; padding: 0.35em 0.625em 0.75em;} legend{box-sizing: border-box; color: inherit; display: table; max-width: 100%; padding: 0; white-space: normal;} progress{display: inline-block; vertical-align: baseline;} textarea{overflow: auto;} [type='checkbox'], [type='radio']{box-sizing: border-box; padding: 0;} [type='number']::-webkit-inner-spin-button, [type='number']::-webkit-outer-spin-button{height: auto;} [type='search']{-webkit-appearance: textfield; outline-offset: -2px;} [type='search']::-webkit-search-cancel-button, [type='search']::-webkit-search-decoration{-webkit-appearance: none;} ::-webkit-file-upload-button{-webkit-appearance: button; font: inherit;} details, menu{display: block;} summary{display: list-item;} canvas{display: inline-block;} template{display: none;} [hidden]{display: none;} html{font-size: 62.5%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;} body{font-size: 1.8rem; line-height: 1.618; max-width: 90em; margin: auto; color: #4a4a4a; background-color: #f9f9f9; padding: 13px;} @media (max-width:684px){body{font-size: 1.53rem;} } @media (max-width:382px){body{font-size: 1.35rem;} } h1, h2, h3, h4, h5, h6{line-height: 1.1; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; font-weight: 700; margin-top: 3rem; margin-bottom: 1.5rem; overflow-wrap: break-word; word-wrap: break-word; -ms-word-break: break-all; word-break: break-word; -ms-hyphens: auto; -moz-hyphens: auto; -webkit-hyphens: auto; hyphens: auto;} h1{font-size: 2.35em;} h2{font-size: 2.00em;} h3{font-size: 1.75em;} h4{font-size: 1.5em;} h5{font-size: 1.25em;} h6{font-size: 1em;} p{margin-top: 0px; margin-bottom: 2.5rem;} small, sub, sup{font-size: 75%;} hr{border-color: #2c8898;} a{text-decoration: none; color: #2c8898;} a:hover{color: #982c61; border-bottom: 2px solid #4a4a4a;} ul{padding-left: 1.4em; margin-top: 0px; margin-bottom: 2.5rem;} li{margin-bottom: 0.4em;} blockquote{font-style: italic; margin-left: 1.5em; padding-left: 1em; border-left: 3px solid #2c8898;} img{height: auto; width: 70%; margin-top: 0px; margin-bottom: 2.5rem;} pre{background-color: #f1f1f1; display: block; padding: 1em; overflow-x: auto; margin-top: 0px; margin-bottom: 2.5rem;} code{font-size: 0.9em; padding: 0 0.5em; background-color: #f1f1f1; white-space: pre-wrap;} pre>code{padding: 0; background-color: transparent; white-space: pre;} table{text-align: justify; width: 100%; border-collapse: collapse;} td, th{padding: 0.5em; border-bottom: 1px solid #f1f1f1;} input, textarea{border: 1px solid #4a4a4a;} input:focus, textarea:focus{border: 1px solid #2c8898;} textarea{width: 100%;} .button, button, input[type='submit'], input[type='reset'], input[type='button']{display: inline-block; padding: 5px 10px; text-align: center; text-decoration: none; white-space: nowrap; background-color: #2c8898; color: #f9f9f9; border-radius: 1px; border: 1px solid #2c8898; cursor: pointer; box-sizing: border-box;} .button[disabled], button[disabled], input[type='submit'][disabled], input[type='reset'][disabled], input[type='button'][disabled]{cursor: default; opacity: .5;} .button:focus, .button:hover, button:focus, button:hover, input[type='submit']:focus, input[type='submit']:hover, input[type='reset']:focus, input[type='reset']:hover, input[type='button']:focus, input[type='button']:hover{background-color: #982c61; border-color: #982c61; color: #f9f9f9; outline: 0;} textarea, select, input[type]{color: #4a4a4a; padding: 6px 10px; margin-bottom: 10px; background-color: #f1f1f1; border: 1px solid #f1f1f1; border-radius: 4px; box-shadow: none; box-sizing: border-box;} textarea:focus, select:focus, input[type]:focus{border: 1px solid #2c8898; outline: 0;} input[type='checkbox']:focus{outline: 1px dotted #2c8898;} label, legend, fieldset{display: block; margin-bottom: .5rem; font-weight: 600;} .out{position: relative; width: 100%; height: 100%; overflow: hidden; display: flex;} .mm{width: 33%; margin: 5px;} .mm img{width: 100%; border-radius: 3px;} .mmi{width: 12.5%; height: auto; margin: 5px;} .mmi img{width: 100%; border-radius: 3px;}</style><style> .listnonema{margin: 0px;} .listnonema label{margin: 0px;} .aright{font-size: 2rem;; position: absolute; right: 30px;}</style><style>.shape-ex1{width: 100%; display: block; transition: all .3s linear;}.shape-ex1 ul{padding-inline-start: 0px; margin: 0rem;}.shape-ex1 li::marker{font-size: 0;}.shape-ex1 input[type='checkbox'], .shape-ex1 input[type='radio']{position: absolute; opacity: 0; pointer-events: none; top: 0;}.shape-ex1-list-box{min-width: max-content; width: 45%;}.shape-ex1-list{position: relative;}.shape-ex1-list-name{font-size: 2.5rem; padding-right: 1.25rem; padding-left: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(209 213 219); display: block; min-width: max-content; position: relative; margin: 0rem;}.shape-ex1-list-name label{margin: 0rem;}.shape-ex1-list-name li{margin: 0rem;}.shape-ex1-list-name::after{content: ''; width: 0; height: 0; border-top: 8px solid #000; border-left: 4px solid transparent; border-right: 4px solid transparent; position: absolute; top: calc(50% - 4px); right: 8px;}.shape-ex1-list-sec-box{display: none;}.shape-ex1-list-sec{}.shape-ex1-list-sec-name{font-size: 2rem; padding-right: 1.25rem; padding-left: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(229 231 235); min-width: max-content; display: block; position: relative;}.shape-ex1-list-sec-name::after{content: ''; width: 0; height: 0; border-top: 8px solid #000; border-left: 4px solid transparent; border-right: 4px solid transparent; position: absolute; top: calc(50% - 4px); right: 6px;}.shape-ex1-list-thr-box{display: none;}.shape-ex1-list-thr-name{font-size: 1.5rem; padding-left: 0.625rem; padding-right: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(243 244 246); min-width: max-content; display: block; position: relative;}.shape-ex1 input[type='checkbox']:checked+ul, .shape-ex1 input[type='radio']:checked+ul{display: block;}</style><style>.huge-btn{width: 45px;height: 45px;position: fixed;border-radius: 10px;}.topbtn{right: 20px;bottom: 20px;display: none;}.topbtnsize{font-size: 30px;}</style></head>"

    con_main_header = "<body class='pagetop' onscroll='get()'><header><h1>Report</h1><h5>Date. " + st + "</h5></header><main>"

    con_bar_h = "<div class='shape-ex1'><ul class='shape-ex1-list-box'><li class='shape-ex1-list listnonema'><label class='shape-ex1-list-name' for='ex1_1-checkbox'>YOLO <a href='#t1' class='aright'>HERE</a></label><input type='checkbox' name='ex1_1-input' id='ex1_1-checkbox' /><ul class='shape-ex1-list-sec-box'><li class='shape-ex1-list-sec listnonema'><label class='shape-ex1-list-sec-name' for='ex1_2_1-checkbox'>Train <a href='#t1w1' class='aright'>HERE</a></label><input type='radio' name='ex1_2-input' id='ex1_2_1-checkbox' /><ul class='shape-ex1-list-thr-box'>"

    con_bar_m = "</ul></li><li class='shape-ex1-list-sec listnonema'><label class='shape-ex1-list-sec-name' for='ex1_2_2-checkbox'>Predict <a href='#t1w2' class='aright'>HERE</a></label><input type='radio' name='ex1_2-input' id='ex1_2_2-checkbox' /><ul class='shape-ex1-list-thr-box'>"

    con_bar_t = "</ul></li></ul></li></ul></div>"

    con_bar_c1 = ""
    con_bar_c2 = ""

    con_main_train_c = ""

    # <img src='data:image/png;base64, ' alt='... ' />
    for filename in os.listdir(train_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # tem_con_main_train_c = "<h4>" + filename + "</h4><p><img src='"+ train_folder + "/" + filename +"' alt='"+filename+"'></p>"
            # Opening an Image File
            image_path = train_folder + "/" + filename # Replace with your image file path
            img = cv2.imread(image_path)
            h,w,_ = img.shape
            img_resize = imutils.resize(img, height=(h//2))
            h,w,_ = img_resize.shape
            print("H : ", h ,"; W : ", w)
            image = cv2.imencode('.png',img_resize)[1]
            base64_head = 'data:image/png;base64,'
            base64_main = str(base64.b64encode(image))[2:-1]
            image_code = base64_head + base64_main
            print("OpenCV resizes the image to base64 length ：%d"%len(image_code))
            # print(image_code)
            tem_con_main_train_c = "<h4 id='t1w1i"+ filename +"'>" + filename + "</h4><p><img src='" + image_code + "' alt='" + filename + "' /></p>"
            tem_con_bar_c1 = "<li class='shape-ex1-list-thr-name listnonema'><span>" + filename + " <a href='#t1w1i" + filename + "' class='aright'>HERE</a></span></li>"
            con_bar_c1 = con_bar_c1 + tem_con_bar_c1
            con_main_train_c = con_main_train_c + tem_con_main_train_c
            print(train_folder + "/" + filename)

# <li class='shape-ex1-list-thr-name listnonema'><span>Image 1 <a href='#t1w1i1' class='aright'>HERE</a></span></li>
# <h4 id='t1w1i2'>Image 2</h4><p><img src='./cow.jpg' alt='cow.jpg'></p>

    con_main_train_h = "<article><header><h1>Info.</h1></header><p>Machine learning image visual segmentation analysis.</p></article><article><header><h1 id='t1'>YOLO</h1></header><p>You Only Look Once (YOLO) is a state-of-the-art, real-time object detection algorithm introduced in 2015 by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi in their famous research paper 'You Only Look Once: Unified, Real-Time Object Detection'. </p></article><article><header><h3 id='t1w1'>Train</h3></header><div><hr></div>"

    con_main_train_t = "</article>"
    # <header><h3 id='t1w2'>Predict</h3></header><div><hr></div>
    con_main_predict_h = "<article><header><h3 id='t1w2'>Predict</h3></header><div><hr></div>"

    files = []
    info_files = []
    files_check = []
    # input_folder = './yolo_runs_202X.../segment/predict'
    input_folder_labels = predict_folder_labels
    for filename in os.listdir(predict_folder_labels):
        if filename.endswith((".txt")):
            info_files.append(predict_folder_labels + "/" + filename)
            files.append(filename) 
            for con in files:
                files_check.append(con.split(".")[0])
    print("INFO. Files - TXT : ", files)
    print("INFO. The File Of Number - TXT : ", len(files))
    print("INFO. TXT Path : ", info_files)

    images = []
    info_images = []
    images_check = []
    for filename in os.listdir(predict_folder):
        if filename.endswith((".png")):
            info_images.append(predict_folder + "/" + filename)
            images.append(filename)
            for con in images:
                images_check.append(con.split(".")[0])
    print("INFO. Images: ", images)
    print("INFO. The Images Of Number : ", len(images))
    print("INFO. Images Path : ", info_images)
    equal_lists = list(set(files_check).intersection(set(images_check)))
    print(equal_lists)
    con_main_predict_unit = ""

    predict_images_path = []
    for filename in equal_lists:
        print(filename)
        image_path = predict_folder + "/" + filename + ".png"
        yplabels_path = predict_folder + "/labels/" + filename + ".txt"
        original_image_path = original_image + "/" + filename + ".png"
        predict_folder_labels_path = predict_folder_labels + "/" + filename + ".png"
        # Original image
        img = cv2.imread(original_image_path)
        h,w,_ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h,w,_ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png',img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        original_image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length ：%d"%len(original_image_code))
        # YOLO2images
        tar = resdir + "/" + filename + ".png"
        img = cv2.imread(tar)
        h,w,_ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h,w,_ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png',img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        yoloraw_image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length ：%d"%len(yoloraw_image_code))
        # Predict
        img = cv2.imread(image_path)
        h,w,_ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h,w,_ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png',img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length : %d"%len(image_code))
        # print(image_code)
        tem_con_bar_c2 = "<li class='shape-ex1-list-thr-name listnonema'><span>" + filename + " <a href='#t1w2i" + filename + "' class='aright'>HERE</a></span></li>"
        con_bar_c2  = con_bar_c2 + tem_con_bar_c2
        tem_con_main_train_c = "<h4 id='t1w2i"+ filename +"'>" + filename + "</h4><p><img src='" + image_code + "' alt='" + filename + "' /></p>"

        tem_con_main_predict_unit = "<h4 id='t1w2i" + filename + "'>"+ filename +"</h4><p><div class='out'><div class='mm'>Original image<br/><img src='" + original_image_code + "' alt='" + filename + " - Original image' /></div><div class='mm'>YOLO label image<br/><img src='" + yoloraw_image_code + "' alt='" + filename + " - YOLO Label Image' /></div><div class='mm'>YOLO training results image<br/><img src='" + image_code + "' alt='" + filename + " - YOLO Training Results Image' /></div></div><!-- IOU : --></p></article>"
        con_main_predict_unit = con_main_predict_unit + tem_con_main_predict_unit
        predict_images_path.append( predict_folder + "/" + filename + ".png")
    # print(predict_images_path)

    # con_main_predict ="<h4>TEST.png</h4><p><div class='out'><div class='mm'>Original image<br/><img src='./TEST.png' alt='Original image'></div><div class='mm'>YOLO label image<br/><img src='./TEST.png' alt='YOLO label image'></div><div class='mm'>YOLO training results image<br/><img src='./TEST.png' alt='YOLO training results image'></div></div><!-- IOU : --></p></article></main>"

    con_bar_all = con_bar_h + con_bar_c1 + con_bar_m + con_bar_c2 + con_bar_t

    con_main_predict_t = "</main>"

    con_main_predict = con_main_predict_h + con_main_predict_unit + con_main_predict_t

    con_main = con_main_header + con_bar_all + con_main_train_h + con_main_train_c + con_main_train_t + con_main_predict

    con_tail = "<footer>"+lt+"</footer></body></html><button class='huge-btn totop topbtn'><b class='topbtnsize'>^</b></button><script> function get(){var topverb1=document.documentElement.scrollTop;var topverb2=document.body.scrollHeight;if(topverb1>=topverb2/8){document.querySelector('.topbtn').style.display='inherit'}else{document.querySelector('.topbtn').style.display='none'}} document.querySelector('.totop').onclick=function(){document.querySelector('.pagetop').scrollIntoView(true)} </script></body></html>"

    con_all = cont_head + con_main + con_tail

    file = open(html_path, 'w')
    file.write(con_all)
    file.close()



# ymal_path = '/mnt/... /dataset.yaml'
ymal_path = args.ypath

# original_image = '/mnt/ ... /yolov8-datasets-predict-name'
original_image = args.original

# predict_folder = './yolo_runs_.../segment/predict'
predict_folder = args.ywork + "/predict"

# train_folder = './yolo_runs_.../segment/train'
train_folder = args.ywork + "/train"

# html_path = "./yolo_runs_.../segment/predict/index.html"
html_path = args.output

# pout = "./yolo_runs_.../yolo2images"
pout = args.yimg


report_function_d(ymal_path, original_image, predict_folder, train_folder, html_path, pout)
