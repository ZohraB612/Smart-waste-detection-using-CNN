# Import necessary packages and libraries
import os
import time
import pathlib
import warnings
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from faster_rcnn_and_ssd.waste.testing import image_boxes_params
from object_detection.utils import visualization_utils as viz_utils
from faster_rcnn_and_ssd.waste.annotations.tfannotations import TFAnnotation
from faster_rcnn_and_ssd.waste.testing.IoU import bb_intersection_over_union

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

FOLDER_DETECTED_IMAGES = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\trained_models_waste_labels_exp2"\
                         r"\faster_rcnn\detected_images\collage"
"""
To run inference for base Faster RCNN model with original labels:
--det_model=faster_rcnn
--config=faster_rcnn_config
--checkpoint=faster_rcnn_ckpt
--labels=coco

For base SSD:
--det_model=ssd
--config=ssd_config
--checkpoint=ssd_ckpt
--labels=coco

For waste trained RCNN:
--det_model=faster_rcnn
--config=faster_rcnn_waste_config
--checkpoint=faster_rcnn_waste_ckpt
--labels=waste

For waste trained SSD:
--det_model=ssd
--config=ssd_waste_config
--checkpoint=ssd_waste_ckpt
--labels=waste
"""

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-det_model", "--det_model",
                type=str, default="faster_rcnn",
                help="name of pre-trained model to use")
ap.add_argument("-config", "--config",
                type=str, default="faster_rcnn_waste_config",
                help="name of pre-trained model's configuration to use")
ap.add_argument("-checkpoint", "--checkpoint",
                type=str, default="faster_rcnn_waste_ckpt",
                help="name of pre-trained model's configuration to use")
ap.add_argument("-labels", "--labels",
                type=str, default="waste",
                help="labels for trained classes in the model")
args = vars(ap.parse_args())

# Model paths
F_RCNN = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\config" \
         r"\faster_rcnn_resnet101_v1_640x640_coco17_tpu-8"
SSD = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\config" \
      r"\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"

# Config paths
F_RCNN_CONFIG = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste" \
                r"\config\faster_rcnn_resnet101_v1_640x640_coco17_tpu-8\pipeline.config"
SSD_CONFIG = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste" \
             r"\config\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8\pipeline.config"

# CKPT paths
F_RCNN_CKPT = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\training\Faster_RCNN\record_5\experiment2"
SS_D_CKPT = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\training\SSD\record_5\experiment2"

# Paths to base model configuration and checkpoint
RCNN_CFG = F_RCNN + r"\pipeline.config"
RCNN_CKPT = F_RCNN + r"\checkpoint"

SSD_CFG = SSD + r"\pipeline.config"
SSD_CKPT = SSD + r"\checkpoint"

MODELS = {
    "faster_rcnn": F_RCNN,
    "ssd": SSD
}

CONFIG = {
    "faster_rcnn_waste_config": F_RCNN_CONFIG,
    "ssd_waste_config": SSD_CONFIG,
    "faster_rcnn_config": RCNN_CFG,
    "ssd_config": RCNN_CKPT,
}

CHECKPOINT = {
    "faster_rcnn_waste_ckpt": F_RCNN_CKPT,
    "ssd_waste_ckpt": SS_D_CKPT,
    "faster_rcnn_ckpt": RCNN_CKPT,
    "ssd_ckpt": SSD_CKPT
}

labels_map = {"cardboard": 0, "glass": 1, "metal": 2, "paper":3, "plastic":4, "trash":5}

# Download labels file for original COCO labels
def download_labels(filename):
    print("download labels")
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)

    label_dir = pathlib.Path(label_dir)
    # print(str(label_dir))
    return str(label_dir)


LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_COCO_LABELS = download_labels(LABEL_FILENAME)
PATH_TO_WASTE_LABELS = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records\classes.pbtxt'

LABELS = {
    "coco": PATH_TO_COCO_LABELS,
    "waste": PATH_TO_WASTE_LABELS
}


def get_local_paths(folder):
    image_paths = []
    for fn in os.listdir(folder):
        image_paths.append(os.path.join(folder, fn))

    return image_paths


# IMAGE_PATHS = get_local_paths(r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\obj_imgs\all_test")
IMAGE_PATHS = get_local_paths(r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\image-collage\new_collage_4")

# Path to TrashNet waste class labels and annotations
# FP_ANNOTATIONS = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\annotations\wasteAnnotations1.csv'
FP_ANNOTATIONS = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\annotations\wasteAnnotations_collages.csv'

PATH_TO_MODEL_DIR = MODELS[args["det_model"]]
CONFIG = CONFIG[args['config']]
CKPT = CHECKPOINT[args['checkpoint']]
LABELS = LABELS[args['labels']]

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint from model
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CKPT, 'ckpt-26')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    det = detection_model.postprocess(prediction_dict, shapes)

    return det


end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(LABELS,
                                                                    use_display_name=True)


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def infer_images(image_paths=IMAGE_PATHS, save_detected_folder=FOLDER_DETECTED_IMAGES):
    i = 1
    a = 0

    # Calculate the total number of detections and no detections
    count_no_detections = 0
    total_num_detections = 0

    # Calculate the number of bad, good, and excellent detections
    bad_detections = 0  # Anything below 0.5
    good_detections = 0  # Anything between 0.5 and 0.8
    excellent_detections = 0  # Anything above 0.8

    tfAnnot = TFAnnotation()
    target_boxes = image_boxes_params.load_image_boxes(FP_ANNOTATIONS)

    for image_path in image_paths:
        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        total_num_detections += num_detections

        # We use this if statement to count how many detections we not found
        if num_detections <= 0:
            count_no_detections += 1

        IMAGE_WIDTH = 512
        IMAGE_HEIGHT = 384

        fn_image = os.path.basename(image_path)
        target_box = target_boxes[fn_image]
        print(target_boxes)
        norm_box = detections['detection_boxes'].numpy()[0, 0, :]

        predicted_box = (norm_box[1] * IMAGE_WIDTH,  # xmin
                         norm_box[0] * IMAGE_HEIGHT,  # ymin
                         norm_box[3] * IMAGE_WIDTH,  # xmax
                         norm_box[2] * IMAGE_HEIGHT)  # ymax

        # Determine Intersection over Union between original annotations boxes and predictions boxes
        # using the function from IoU file

        for t_box in target_box :

            iou = bb_intersection_over_union(t_box, predicted_box)
            print(iou)

            # These if statements are used to count the number of bad, good,
            # and excellent detections using the Intersection over Union
            if iou < 0.5:
                bad_detections += 1
            elif 0.5 <= iou < 0.8:
                good_detections += 1
            elif iou >= 0.8:
                excellent_detections += 1

        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # print('this is det class', detections['detection_classes'])
        # We use the label_id_offset to shift the class labels by 1
        # This is done so that the model receives on-hot labels where non-background
        # classes start counting at the zeroth index
        label_id_offset = 1

        # The numpy array images are copied to apply detection boxes on them
        image_np_with_detections = image_np.copy()

        # The following function is used to visualise the boxes, classes, and
        # scores for the detections made by the model
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=3,
            min_score_thresh=.30,
            agnostic_mode=False)

        # On the same image, we plot the original annotation box for the waste items
        fix, ax = plt.subplots()
        ax.imshow(image_np_with_detections)

        for t_box in target_box:
            corner = (t_box[0], t_box[1])
            box_w = t_box[2] - t_box[0]
            box_h = t_box[3] - t_box[1]
            rect = patches.Rectangle(corner, box_w, box_h,
                                     edgecolor="b",
                                     linewidth=2,
                                     facecolor="none",
                                     fill=False)

            res = image_boxes_params.load_image_boxes(FP_ANNOTATIONS)
            # print(f'this is res {res[image_path][4]}')

            font = {'weight' : 'bold',
                    'size' : 22,
                    'color' : 'blue'}
            plt.text(t_box[0], t_box[1], t_box[4], fontdict=font)
            ax.add_patch(rect)

            pred_lbl = detections['detection_classes'] + label_id_offset
            if len(pred_lbl) != 0 and labels_map[t_box[4]] in pred_lbl:
                a += 1

        # These images are then saved in the testing folder
        fp = (r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\trained_models_waste_labels_exp2"
              r"\faster_rcnn\pred_images\collage\\" + fn_image)
        plt.savefig(fp)

        if save_detected_folder != "":
            xmin, ymin, xmax, ymax = [int(n) for n in predicted_box]
            area = (xmax - xmin) * (ymax - ymin)
            if area < IMAGE_WIDTH * IMAGE_HEIGHT * 0.1:  # less than 10% of the image??
                print(f"{fn_image} is too small")
            else:
                cropped_img = image_np[xmin:xmax, ymin:ymax, :]

                # fix, ax = plt.subplots()
                # ax.imshow(cropped_img,)

                from PIL import Image
                img = Image.fromarray(cropped_img.astype("uint8"), "RGB")

                fp = os.path.join(save_detected_folder, fn_image)
                # plt.savefig(fp)
                img.save(fp)

        i += 1

    print('Inference process is done!')

    print(f"NOTHING DETECTED: {count_no_detections}, TOTAL DETECTIONS: {total_num_detections}")
    print(f"Number of Bad detections: {bad_detections}, Good detections: {good_detections}, "
          f"Excellent detections: {excellent_detections}")
    print(f"Number of correctly labelled items {a}")


if __name__ == "__main__":
    infer_images(IMAGE_PATHS, FOLDER_DETECTED_IMAGES)

# Links
# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# https://www.tensorflow.org/hub/tutorials/object_detection
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html
# https://stackoverflow.com/questions/58994588/how-can-i-add-the-counts-to-the-histogram-plot
