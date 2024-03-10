import os
import cv2
import shutil
from tqdm import tqdm


CLASS_NAMES = {'Car': 0, 'Van':1, 'Truck':2, 'Pedestrian':3, \
               'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}


image_path = '/AI/img'
label_path = '/AI/label'

new_train_img_path = 'kitti_yolo/images/train'
new_test_img_path = 'kitti_yolo/images/test'
new_train_lab_path = 'kitti_yolo/labels/train'
new_test_lab_path = 'kitti_yolo/labels/test'

os.makedirs(new_train_img_path, exist_ok=True)
os.makedirs(new_test_img_path, exist_ok=True)
os.makedirs(new_train_lab_path, exist_ok=True)
os.makedirs(new_test_lab_path, exist_ok=True)


files_list = sorted(os.listdir(image_path))
files_list = [file[:-4] for file in files_list]

train_ratio = int(0.90 * len(files_list))


for i in tqdm(range(len(files_list))): 
    h, w = 370, 1224

    with open(f'{label_path}/{files_list[i]}.txt', "r") as file:
        lines = file.readlines()

        yolo_labels = []
        for line in lines:
            data = line.strip().split()

            if data[0] == 'DontCare':
                continue
            
            label = CLASS_NAMES[data[0]]

            center_x = ((float(data[6]) + float(data[4]))/2) / w
            center_y = ((float(data[7]) + float(data[5]))/2) / h

            width = (float(data[6]) - float(data[4])) / w
            height = (float(data[7]) - float(data[5])) / h

            yolo_labels.append((label, center_x, center_y, width, height))
      

    if i < train_ratio:
        shutil.copyfile(image_path + '/' + files_list[i] + '.png', new_train_img_path + '/' + files_list[i] + '.png')

        with open(f'{new_train_lab_path}/{files_list[i]}.txt', "w") as lab_data:
            for item in yolo_labels:
                lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")
                
    else:
        shutil.copyfile(image_path + '/' + files_list[i] + '.png', new_test_img_path + '/' + files_list[i] + '.png')

        with open(f'{new_test_lab_path}/{files_list[i]}.txt', "w") as lab_data:
            for item in yolo_labels:
                lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")
