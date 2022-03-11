import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

base_path = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste'
annotation_path = os.path.sep.join([base_path, 'images/image-collage/xml_collage_4']).replace("\\", "/")

cols = ["filename", "width", "height", "depth", "xmin", "ymin", "xmax", "ymax", "name"]
rows = []

# path_all_train = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\obj_imgs\all_train"
path_collages = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\image-collage\new_collage_4"

for filename in os.listdir(annotation_path):
    if not filename.endswith('.xml'):
        continue
    filename_noxml = os.path.splitext(filename)[0]
    # if os.path.isfile(path_all_train + "/" + filename_noxml + ".jpg"):
    if os.path.isfile(path_collages + "/" + filename_noxml + ".png"):
        fullname = os.path.join(annotation_path, filename)
        tree = ET.parse(fullname)

        root = tree.getroot()
        size = root.find("size")

        objectEls = root.findall('object')

        for objectEl in objectEls:
            bndboxEl = objectEl.find("bndbox")

            fName = root.find("filename").text

            width = size.find("width").text
            height = size.find("height").text
            depth = size.find("depth").text

            xmin = bndboxEl.find("xmin").text
            ymin = bndboxEl.find("ymin").text
            xmax = bndboxEl.find("xmax").text
            ymax = bndboxEl.find("ymax").text
            name = objectEl.find("name").text
            difficult = objectEl.find("difficult").text

            rows.append({"filename": fName,
                         "width": width,
                         "height": height,
                         "depth": depth,
                         "xmin": xmin,
                         "ymin": ymin,
                         "xmax": xmax,
                         "ymax": ymax,
                         "name": name,
                         "difficult": difficult})

df = pd.DataFrame(rows, columns=cols)

# df.to_csv("C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/annotations/wasteAnnotations_all_train.csv")
df.to_csv("C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/annotations/wasteAnnotations_collages.csv")
