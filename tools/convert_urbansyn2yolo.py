import os 
import json
import shutil
import cv2

CLASSES = {'bicycle': 0, 'bus':1, 'car':2, 'motorcycle':3, 'person':4, 
           'rider':5, 'train':6, 'truck':7}

image_width = int(2048)
image_height = int(1024)


if __name__ == "__main__":
    dataset_path = '/AI'
    
    os.makedirs(os.path.join(dataset_path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'images'), exist_ok=True)

    for jfile in sorted(os.listdir(os.path.join(dataset_path, 'bbox2d'))):
        bbox = []
        
        with open(os.path.join(dataset_path, 'bbox2d', jfile), 'r') as file:
            data = json.load(file)
        
        # img = cv2.imread(f"{dataset_path}/syndet_datasets/urbansyn_yolo/images/train/{jfile.split('.')[0].split('_')[-1]}.png")  
                      
        for dat in data:
            if dat['label'] == 'car':
                clss = 0
            elif dat['label'] == 'truck':
                clss = 1
            elif dat['label'] == 'person':
                clss = 2
            elif dat['label'] == 'rider':
                clss = 3
            else:
                continue    
            
            x_center = ((float(dat['bbox']['xMax']) + float(dat['bbox']['xMin'])) / 2 ) / image_width
            y_center = ((float(dat['bbox']['yMax']) + float(dat['bbox']['yMin'])) / 2 ) / image_height
            width = (float(dat['bbox']['xMax']) - float(dat['bbox']['xMin'])) / image_width
            height = (float(dat['bbox']['yMax']) - float(dat['bbox']['yMin'])) / image_height
            
            if int(dat['bbox']['xMax']) > image_width or int(dat['bbox']['xMin'])< 0 or int(dat['bbox']['yMax']) > image_height or int(dat['bbox']['yMin']) < 0:
                print(f"xmax: {dat['bbox']['xMax']}, xmin: {dat['bbox']['xMin']}, ymax: {dat['bbox']['yMax']}, ymin: {dat['bbox']['yMin']}")
            
            bbox.append((clss, x_center, y_center, width, height))
    
            # cv2.rectangle(img, (int(dat['bbox']['xMin']), int(dat['bbox']['yMin'])), (int(dat['bbox']['xMax']), int(dat['bbox']['yMax'])), (255,0,0), 2, 1)
            
        # cv2.imshow('dada', img)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()
            
        if len(bbox) >= 1:
            fl = jfile.split('.')[0].split('_')[-1]

            with open(f'{dataset_path}/labels/{fl}.txt', "w") as lab_data:
                        for item in bbox:
                            lab_data.write(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}' + "\n")            
                        
        
            shutil.copyfile(f"{dataset_path}/rgb/rgb_{fl}.png", f"{dataset_path}/images/{fl}.png")    
