import os
import cv2
import shutil
from tqdm import tqdm


CLASS_NAMES = {'Car': 0, 'Van':1, 'Truck':2, 'Pedestrian':3, \
               'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}


image_path = os.path.expanduser('~') + '/synDet/datasets/kitti/images'
label_path = os.path.expanduser('~') + '/syndet-yolo-dcan/datasets/kitti/labels'

new_train_img_path = 'kitti_yolo/images/train'
new_test_img_path = 'kitti_yolo/images/test'
new_train_lab_path = 'kitti_yolo/labels/train'
new_test_lab_path = 'kitti_yolo/labels/test'

os.makedirs(new_train_img_path, exist_ok=True)
os.makedirs(new_test_img_path, exist_ok=True)
os.makedirs(new_train_lab_path, exist_ok=True)
os.makedirs(new_test_lab_path, exist_ok=True)


files_list = os.listdir(image_path)
files_list = [file[:-4] for file in files_list]

train_ratio = int(0.80 * len(files_list))

for i in tqdm(range(len(files_list))): 
    img = cv2.imread(f'{image_path}/{files_list[i]}.png')

    h, w = img.shape[0], img.shape[1]

    try:
        with open(f'{label_path}/{files_list[i]}.txt', "r") as file:
            lines = file.readlines()

            yolo_labels = []

            for line in lines:
                data = line.strip().split()

                if data[0] == 'Car':
                    label = 0
                elif data[0] == 'Truck':
                    label = 1
                elif data[0] == 'Pedestrian':
                    label = 2
                elif data[0] == 'Cyclist':
                    label = 3

                center_x = ((float(data[6]) + float(data[4]))/2) / w
                center_y = ((float(data[7]) + float(data[5]))/2) / h

                width = (float(data[6]) - float(data[4])) / w
                height = (float(data[7]) - float(data[5])) / h

                try:
                    yolo_labels.append((label, center_x, center_y, width, height))
                except:
                    continue

        if yolo_labels is not None:
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
    except:
        continue