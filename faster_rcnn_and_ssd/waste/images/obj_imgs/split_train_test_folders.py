import os
import shutil
from random import shuffle

BASE_PATH = 'C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images/imgs'
folder_test = 'C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images/obj_imgs/test'
folder_train = 'C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images/obj_imgs/train'

for folder_name in os.listdir(BASE_PATH):
    folder_len = len(os.listdir(BASE_PATH + '/' + folder_name)) * 0.2
    shuffle(os.listdir(BASE_PATH + '/' + folder_name))
    for img in os.listdir(BASE_PATH + '/' + folder_name):
        file_path_type = os.path.join(
            "C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images/imgs/" + folder_name + '/' + img)

        # print(folder_name)
        # print(len(os.listdir(folder_test + '/' + folder_name)) )
        # print(len(os.listdir(BASE_PATH + '/' + folder_name)))
        # print(len(os.listdir(BASE_PATH + '/' + folder_name)) * 0.2)
        if folder_name == 'cardboard' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        elif folder_name == 'paper' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        elif folder_name == 'glass' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        elif folder_name == 'plastic' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        elif folder_name == 'trash' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        elif folder_name == 'metal' and len(os.listdir(folder_test + '/' + folder_name)) <= folder_len:
            shutil.move(file_path_type, folder_test + '/' + folder_name)
        else:
            shutil.move(file_path_type, folder_train + '/' + folder_name)

