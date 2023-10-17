import os
import cv2
import shutil
from tqdm import tqdm
from nuimages import NuImages


nuim = NuImages(dataroot=os.path.expanduser('~') + '/syndet-yolo-dcan/datasets/nuimages', version='v1.0-train', verbose=True, lazy=True)

image_path = os.path.expanduser('~') + '/syndet-yolo-dcan/datasets/nuimages'


new_train_img_path = 'nuimages_yolo/images/train'
new_test_img_path = 'nuimages_yolo/images/test'
new_train_lab_path = 'nuimages_yolo/labels/train'
new_test_lab_path = 'nuimages_yolo/labels/test'

os.makedirs(new_train_img_path, exist_ok=True)
os.makedirs(new_test_img_path, exist_ok=True)
os.makedirs(new_train_lab_path, exist_ok=True)
os.makedirs(new_test_lab_path, exist_ok=True)

print(nuim.category[0])
print(nuim.table_names)
print(nuim.__dir__())

train_ratio = int(0.80 * len(nuim.sample))

for i in tqdm(range(len(nuim.sample))):
    try:
        token = nuim.sample[i]['token']
        object_tokens, surface_tokens = nuim.list_anns(token)

        img_data = nuim.get('sample_data', nuim.get('object_ann', object_tokens[0])['sample_data_token'])
        img_name = img_data['filename']
        img_w = img_data['width']
        img_h = img_data['height']

        img = cv2.imread(image_path + '/' + img_name)

        yolo_labels = []
        for objtok in object_tokens:
            data = nuim.get('object_ann', objtok)

            labels = nuim.get('category', data['category_token'])['name']
            if labels.split('.')[0] == 'human':
                label = 2
            elif labels.split('.')[0] == 'vehicle' and labels.split('.')[1] == 'bicycle':
                label = 3
            elif labels.split('.')[0] == 'vehicle' and labels.split('.')[1] == 'car':
                label = 0

            elif labels.split('.')[0] == 'vehicle' and labels.split('.')[1] == 'truck':
                label = 1
            
            xmin = int(data['bbox'][0])
            ymin = int(data['bbox'][1])
            xmax = int(data['bbox'][2])
            ymax = int(data['bbox'][3])
            
            center_x = ((xmax + xmin) / 2) / img_w
            center_y = ((ymax + ymin) / 2) / img_h
            width =  (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            try:
                yolo_labels.append((label, center_x, center_y, width, height))
            except:
                continue

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        
        if yolo_labels is not None:
            if i < train_ratio: 
                shutil.copyfile(image_path + '/' + img_name, new_train_img_path + '/' + img_name.split('/')[-1])
                with open(f"{new_train_lab_path}/{img_name.split('/')[-1][:-4]}.txt", "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")
            else:
                shutil.copyfile(image_path + '/' + img_name, new_test_img_path + '/' + img_name.split('/')[-1])
                with open(f"{new_test_lab_path}/{img_name.split('/')[-1][:-4]}.txt", "w") as lab_data:
                    for item in yolo_labels:
                        lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")
            

        """ cv2.imshow("sssss", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows() """
    except:
        continue