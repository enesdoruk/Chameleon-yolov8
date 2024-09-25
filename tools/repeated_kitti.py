import os 
import shutil
from tqdm import tqdm




def create_and_move(num_repeated):
    print("------------   Creating Data   ------------")
    for i in tqdm(range(num_repeated)):
        for img in os.listdir(os.path.join(dataset_path, 'images', 'train')):
            shutil.copy(os.path.join(dataset_path, 'images', 'train', img), 
                        os.path.join(dataset_path, 'new_data', 'img', f'repeated_{i}_{img}'))
            
            txt_name = img.split('.')[0]
            shutil.copy(os.path.join(dataset_path, 'labels', 'train', f"{txt_name}.txt"), 
                        os.path.join(dataset_path, 'new_data', 'label', f'repeated_{i}_{txt_name}.txt'))
            
    print("------------   Moving Data   ------------")
    for img in os.listdir(os.path.join(dataset_path, 'new_data', 'img')):
            shutil.move(os.path.join(dataset_path, 'new_data', 'img', img), 
                        os.path.join(dataset_path, 'images', 'train', img))
            
            txt_name = img.split('.')[0]
            shutil.move(os.path.join(dataset_path, 'new_data', 'label', f"{txt_name}.txt"), 
                        os.path.join(dataset_path, 'labels', 'train', f'{txt_name}.txt'))
            
if __name__ == "__main__":
    dataset_path = '/AI/syndet_datasets/kitti_yolo'

    num_repeated = 25

    os.makedirs(os.path.join(dataset_path, 'new_data', 'img'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'new_data', 'label'), exist_ok=True)
    
    create_and_move(num_repeated)