import argparse

import cv2
import numpy as np
import argparse
from faster_rcnn_and_ssd.waste.config import config
from faster_rcnn_and_ssd.waste.annotations.tfannotations import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-label_per", "--label_per",
                type=float, default=0.05,
                help="percentage of labeled data")
ap.add_argument("-category", "--category",
                type=str, default='trash',
                help="class name")
args = vars(ap.parse_args())


def main(_):
    print(config.classes_file)
    f = open(config.classes_file, "w")
    print(f)

    for (k, v) in config.classes.items():
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                                    "\tname: '" + k + "'\n"
                                                      "}\n")
        f.write(item)
    f.close()

    D = {}
    rows = open(config.annotation_path).read().strip().split("\n")  #

    for row in rows[1:]:
        # print(row.split(","))
        row = row.split(",")
        (_, filename, width, height, depth, xmin, ymin, xmax, ymax, label) = row
        print(label)
        if label != args['category']:
            continue

        # print(config.base_path)
        (xmin, ymin) = (float(xmin), float(ymin))
        (xmax, ymax) = (float(xmax), float(ymax))

        if label not in config.classes:
            continue

        p = os.path.sep.join([config.image_path, filename]).replace("\\", "/")
        b = D.get(p, [])

        b.append((label, (xmin, ymin, xmax, ymax)))
        D[p] = b

    (trainKeys, valKeys) = train_test_split(list(D.keys()),
                                            train_size=config.train_size,
                                            test_size=config.valid_size,
                                            random_state=42)

    # (trainKeys, valKeys) = train_test_split(trainKeys,
    #                                         train_size=config.train_size,
    #                                         test_size=config.valid_size,
    #                                         random_state=42)

    temp_train_record = config.train_record
    temp_test_record = config.valid_record
    print('this is args label', args['label_per'])
    datasets = [
        ("train", trainKeys, temp_train_record.replace('.', args['category'] + str(args['label_per']) + '.')),
        # ("validation", valKeys, config.valid_record),
        ("validation", valKeys, temp_test_record.replace('.', args['category'] + str(args['label_per']) + '.'))
    ]

    for (dType, keys, outputPath) in datasets:
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.io.TFRecordWriter(outputPath)
        total = 0

        labeled, unlabeled = np.split(keys, [int(len(keys) * args['label_per'])])
        print(len(keys))
        print(len(labeled))
        for k in labeled:
            encoded = tf.io.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            for (label, (xmin, ymin, xmax, ymax)) in D[k]:
                xmin = xmin / w
                xmax = xmax / w
                ymin = ymin / h
                ymax = ymax / h

                # image = cv2.imread(k)
                # xmin = int(xmin * w)
                # ymin = int(ymin * h)
                # xmax = int(xmax * w)
                # ymax = int(ymax * h)
                #
                # cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                #               (0, 255, 0), 2)
                # cv2.imshow("Image", image)
                # cv2.waitKey(0)

                tfAnnot.xmin.append(xmin)
                tfAnnot.xmax.append(xmax)
                tfAnnot.ymin.append(ymin)
                tfAnnot.ymax.append(ymax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.classes[label])
                tfAnnot.difficult.append(0)

                total += 1

            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total, dType))


if __name__ == "__main__":
    tf.compat.v1.app.run()
