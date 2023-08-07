import os 

train_img = os.listdir('shift_yolo/images/train')
test_img = os.listdir('shift_yolo/images/test')

new_len_train = len(train_img) // 1.5
new_len_test = len(test_img) // 2

for i in range(len(train_img)):
    txt = f'shift_yolo/labels/train/{train_img[i][:-4]}.txt'
    img = f'shift_yolo/images/train/{train_img[i]}'
    
    os.remove(img)
    os.remove(txt)

    if len(os.listdir('shift_yolo/images/train')) == new_len_train:
        break

""" for j in range(len(test_img)):
    txt = f'shift_yolo/labels/test/{test_img[j][:-4]}.txt'
    img = f'shift_yolo/images/test/{test_img[j]}'
    
    os.remove(img)
    os.remove(txt)

    if len(os.listdir('shift_yolo/images/test')) == new_len_test:
        break """