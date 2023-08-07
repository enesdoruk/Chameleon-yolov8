import os 
import cv2
import shutil
from tqdm import tqdm


image_path = os.path.expanduser('~') + '/yolov8/datasets/presil/img'
label_path = os.path.expanduser('~') + '/yolov8/datasets/presil/lab'

CLASS_NAMES = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Truck': 3, 'Person_sitting': 4, 'Motorbike': 5, 'Trailer': 6, \
               'Bus': 7, 'Railed': 8, 'Airplane': 9, 'Boat': 10, 'Animal': 11, 'DontCare': 12, 'Misc': 13, 'Van': 14, 'Tram': 15}


new_train_img_path = 'presil_yolo/images/train'
new_test_img_path = 'presil_yolo/images/test'
new_train_lab_path = 'presil_yolo/labels/train'
new_test_lab_path = 'presil_yolo/labels/test'

os.makedirs(new_train_img_path, exist_ok=True)
os.makedirs(new_test_img_path, exist_ok=True)
os.makedirs(new_train_lab_path, exist_ok=True)
os.makedirs(new_test_lab_path, exist_ok=True)


img_list = os.listdir(image_path)
img_list = [img[:-4] for img in img_list] 

train_ratio = int(0.80 * len(img_list))

for i in tqdm(range(len(img_list))):
    try:
        yolo_labels = []

        img = cv2.imread(f'{image_path}/{img_list[i]}.png')
        try:
            H, W = img.shape[0], img.shape[1]
        except:
            continue

        with open(f'{label_path}/{img_list[i]}.txt', 'r') as txt:
            data = [lin.strip().split() for lin in txt]

        for line in data:
            if line[0] == 'Car':
                label = 0
            elif line[0] == 'Truck':
                label = 1
            elif line[0] == 'Pedestrian':
                label = 2
            elif type == 'Cyclist':
                line[0] = 3
            
            xmin = int(line[4])
            ymin = int(line[5])
            xmax = int(line[6])
            ymax = int(line[7])

            center_x = ((xmax + xmin) / 2) / W
            center_y = ((ymax + ymin) / 2) / H
            width = (xmax - xmin) / W
            height = (ymax - ymin) / H

            try:
                yolo_labels.append((label, center_x, center_y, width, height))
            except:
                continue
        
        if yolo_labels is not None:
            if i < train_ratio:
                with open(f"{new_train_lab_path}/{img_list[i]}.txt", "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")

                shutil.copyfile(f"{image_path}/{img_list[i]}.png", f"{new_train_img_path}/{img_list[i]}.png")    
            else:
                with open(f"{new_test_lab_path}/{img_list[i]}.txt", "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")

                shutil.copyfile(f"{image_path}/{img_list[i]}.png", f"{new_test_img_path}/{img_list[i]}.png")    
    except:
        continue