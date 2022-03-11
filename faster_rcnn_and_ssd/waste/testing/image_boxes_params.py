# Import necessary packages
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

# Path to annotation CSV file
filename = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\annotations\wasteAnnotations.csv'


def load_image_boxes(filename):
    res = {}
    with open(filename, newline='') as new_file:
        reader = csv.reader(new_file)
        print("---")
        # We loop through the csv file using the reader
        # and save the filename, xmin, ymin, xmax, ymax
        # and label as lists
        for i, items in enumerate(reader):
            if i == 0:
                continue
            filename = items[1]
            label = items[9]
            xmin = int(items[5])
            ymin = int(items[6])
            xmax = int(items[7])
            ymax = int(items[8])
            # For collage, uncomment this to be able to draw and
            # access more than one bounding box per image
            if filename in res:
                res[filename].append((xmin, ymin, xmax, ymax, label))
            else:
                res[filename] = [(xmin, ymin, xmax, ymax, label)]
            # Uncomment this for other use (single object images)
            # res[filename] = (ymin, xmin, ymax, xmax, label)

    return res


def load_image_labels(filename):
    res = {}
    with open(filename, newline='') as new_file:
        reader = csv.reader(new_file)
        print("---")
        # We loop through the csv file using the reader
        # and save the filename, xmin, ymin, xmax, ymax
        # and label as lists
        for i, items in enumerate(reader):
            if i == 0:
                continue
            filename = items[1]
            label = items[9]
            # For collage, uncomment this to be able to draw and
            # access more than one bounding box per image
            if filename in res:
                res[filename].append(label)
            else:
                res[filename] = [label]

    return res

