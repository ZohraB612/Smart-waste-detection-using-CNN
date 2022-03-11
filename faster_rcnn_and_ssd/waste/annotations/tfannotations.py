# Import packages
from faster_rcnn_and_ssd.models.research.object_detection.utils.dataset_util import bytes_list_feature, float_list_feature, \
    int64_list_feature, int64_feature, bytes_feature
import tensorflow as tf


class TFAnnotation:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.textLabels = []
        self.classes = []
        self.difficult = []

        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.encoding.encode("utf8"))
        image = bytes_feature(self.image)
        xmin = float_list_feature(self.xmin)
        xmax = float_list_feature(self.xmax)
        ymin = float_list_feature(self.ymin)
        ymax = float_list_feature(self.ymax)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)

        data = {
            "image/height": h,
            "image/width": w,
            "image/filename": filename,
            "image/source_id": filename,
            "image/encoded": image,
            "image/format": encoding,
            "image/object/bbox/xmin": xmin,
            "image/object/bbox/xmax": xmax,
            "image/object/bbox/ymin": ymin,
            "image/object/bbox/ymax": ymax,
            "image/object/class/text": textLabels,
            "image/object/class/label": classes,
            "image/object/difficult": difficult
        }

        return data
