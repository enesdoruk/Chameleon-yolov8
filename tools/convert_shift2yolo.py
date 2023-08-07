import os
import json
import shutil
from tqdm import tqdm


CLASS_NAMES = {'pedestrian':0, 'car':1, 'truck':2, 'bus':3, 'motorcycle':4, 'bicycle':5}

image_path = os.path.expanduser('~') + '/syndet-yolo/datasets/SHIFT/images'
label_path = os.path.expanduser('~') + '/syndet-yolo/datasets/SHIFT'

W = 1280 
H = 800

new_train_img_path = 'shift_yolo/images/train'
new_test_img_path = 'shift_yolo/images/test'
new_train_lab_path = 'shift_yolo/labels/train'
new_test_lab_path = 'shift_yolo/labels/test'

os.makedirs(new_train_img_path, exist_ok=True)
os.makedirs(new_test_img_path, exist_ok=True)
os.makedirs(new_train_lab_path, exist_ok=True)
os.makedirs(new_test_lab_path, exist_ok=True)


with open(label_path + '/' +'label.json') as f:
    data = json.load(f)

print("Keys = ", data.keys())
print(data['frames'][1].keys())
print("Config = ", data['config'])
print("Len of dataset = ", len(data['frames']))

train_ratio = int(0.80 * len(data['frames']))

for i in tqdm(range(len(data['frames']))):
    try:
        img_name = data['frames'][i]['name']
        folder_name = data['frames'][i]['videoName']
        
        yolo_labels = []
        for label in data['frames'][i]['labels']:
            type = label['category']
            if type == 'car':
                lab = 0
            elif type == 'truck':
                lab = 1
            elif type == 'pedestrian':
                lab = 2
            elif type == 'bicycle':
                lab = 3
            
            
            xmin = float(label['box2d']['x1'])
            ymin = float(label['box2d']['y1'])
            xmax = float(label['box2d']['x2'])
            ymax = float(label['box2d']['y2'])

            center_x = ((xmax + xmin) / 2) / W
            center_y = ((ymax + ymin) / 2) / H
            width = (xmax - xmin) / W
            height = (ymax - ymin) / H

            try:
                yolo_labels.append((lab , center_x, center_y, width, height))
            except:
                continue

        if yolo_labels is not None:
            if i < train_ratio:
                shutil.copyfile(f"{image_path}/{folder_name}/{img_name}", f"{new_train_img_path}/{folder_name}_{img_name}")
                
                with open(f'{new_train_lab_path}/{folder_name}_{img_name[:-4]}.txt', "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")
            else:
                shutil.copyfile(f"{image_path}/{folder_name}/{img_name}", f"{new_test_img_path}/{folder_name}_{img_name}")
                
                with open(f'{new_test_lab_path}/{folder_name}_{img_name[:-4]}.txt', "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")

    except:
        continue