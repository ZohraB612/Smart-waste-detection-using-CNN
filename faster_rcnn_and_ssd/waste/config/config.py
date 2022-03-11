import os
import cv2

base_path = 'C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste'
annotation_path = os.path.sep.join([base_path, 'annotations/wasteAnnotations_all_train.csv']).replace("\\", "/")
image_path = os.path.sep.join([base_path, 'images/obj_imgs/all_train']).replace("\\", "/")

# print(base_path)
# suffix = "-glass"
# train_record = os.path.sep.join([base_path, f"records/training{suffix}.record"]).replace("\\", "/")
# valid_record = os.path.sep.join([base_path, f"records/validation{suffix}.record"]).replace("\\", "/")
# classes_file = os.path.sep.join([base_path, f"records/classes{suffix}.pbtxt"]).replace("\\", "/")

train_record = os.path.sep.join([base_path, f"records/training.record"]).replace("\\", "/")
valid_record = os.path.sep.join([base_path, f"records/validation.record"]).replace("\\", "/")
classes_file = os.path.sep.join([base_path, f"records/classes.pbtxt"]).replace("\\", "/")

# print(classes_file)
classes = {"cardboard": 1, "glass": 2, "metal": 3, "paper": 4, "plastic": 5, "trash": 6}
# classes = {"glass": 1}
train_size = 0.8
valid_size = 1 - train_size
