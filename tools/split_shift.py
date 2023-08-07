import os
import random
import shutil
from tqdm import tqdm


image_path = 'img'
label_path = 'lab'

train_img_path = 'images/train'
test_img_path = 'images/test'
train_label_path = 'labels/train'
test_label_path = 'labels/test'

os.makedirs(train_img_path, exist_ok=True)
os.makedirs(test_img_path, exist_ok=True)
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(test_label_path, exist_ok=True)

data_list = os.listdir(image_path)
data_list = [data[:-4] for data in data_list]

random.shuffle(data_list)

train_ratio = 0.80

for i in tqdm(range(len(data_list))):
    if i < train_ratio * len(data_list):
        shutil.copyfile(f"{image_path}/{data_list[i]}.jpg", f"{train_img_path}/{data_list[i]}.jpg")
        shutil.copyfile(f"{label_path}/{data_list[i]}.txt", f"{train_label_path}/{data_list[i]}.txt")
    else:
        shutil.copyfile(f"{image_path}/{data_list[i]}.jpg", f"{test_img_path}/{data_list[i]}.jpg")
        shutil.copyfile(f"{label_path}/{data_list[i]}.txt", f"{test_label_path}/{data_list[i]}.txt")