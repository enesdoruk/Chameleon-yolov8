
import os 
from tqdm import tqdm
import xml.etree.ElementTree as ET

path = '/home/adastec/syndet_datasets/sim10k/labels'
save_path = '/home/adastec/syndet_datasets/sim10k/yolo_labels'
files = os.listdir(path)

os.makedirs(save_path, exist_ok=True)

classes = {'person':2, 'motorbike':1, 'car':0}

for file in tqdm(files):
    tree = ET.parse(os.path.join(path, file))
    root = tree.getroot()
    
    img_name = file.split('.')[0]
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    
    yolo_labels = []
    for object_elem in root.findall('.//object'):
        name = object_elem.find('name').text
        xmin = int(object_elem.find('.//bndbox/xmin').text)
        ymin = int(object_elem.find('.//bndbox/ymin').text)
        xmax = int(object_elem.find('.//bndbox/xmax').text)
        ymax = int(object_elem.find('.//bndbox/ymax').text)
        
        center_x = ((xmax + xmin) / 2) / width
        center_y = ((ymax + ymin) / 2) / height
        width =  (xmax - xmin) / width
        height = (ymax - ymin) / height
        label = classes[name]
        
        try:
            yolo_labels.append((label, center_x, center_y, width, height))
        except:
            continue
    
    with open(f"{save_path}/{img_name}.txt", "w") as lab_data:
        for item in yolo_labels:
            lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")


