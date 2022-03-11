# We start by importing the necessary Python packages.
# Note that most packages are part of the Keras library.
import os
import pathlib
import cv2.cv2
import keras
import argparse
import numpy as np
import tensorflow as tf
from mlxtend.evaluate import mcnemar
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.utils import class_weight
from keras.metrics import CategoricalAccuracy
from keras.layers import GlobalAvgPool2D, Dense
from keras.losses import CategoricalCrossentropy
from keras_preprocessing.image import ImageDataGenerator
# from statsmodels.stats.contingency_tables import mcnemar
from faster_rcnn_and_ssd.waste.testing import image_boxes_params
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    classification_report
from tensorflow.keras.applications import Xception, InceptionV3, VGG16, \
    VGG19, ResNet50, imagenet_utils, EfficientNetB7, ResNet101

from faster_rcnn_and_ssd.waste.testing.inference import FP_ANNOTATIONS, \
    get_local_paths, FOLDER_DETECTED_IMAGES, load_image_into_numpy_array
from faster_rcnn_and_ssd.waste.testing.image_boxes_params import *


LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Set up the number of classes, and
# the size of the train, validation and test sets.
NUM_CLASSES = 6
TRAIN_SIZE = 0.6
VAL_SIZE = 0.1
TEST_SIZE = 0.2

NUM_EPOCHS = 30
NUM_EPOCHS_FINE_TUNE = 30

# Folder where models are saved
ROOT_FOLDER = r"C:\Users\Zohra\MScProject\classification\fine_tuned_models"
MODEL_FOLDER = r"C:\Users\Zohra\MScProject\classification\base_models\my_EfficientNet_model"

# As well as setting up the directories for the training and testing sets
TRAIN_DIREC = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\obj_imgs\train"
TEST_DIREC = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\obj_imgs\test"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model",
                type=str, default="efficientnet",
                help="name of pre-trained network to use")
args = vars(ap.parse_args())

# After constructing the argument parse, we
# define a dictionary that maps model names to their classes
# inside Keras.
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "efficientnet": EfficientNetB7
}

# The following line with help ensure that
# a valid model name was supplied via command line argument.
# If the 'model' name is not found inside MODELS, an AssertionError will be raised.
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
                         "be a key in the `MODELS` dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224, 3)
img_size = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image pre-processing function
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299, 3)
    img_size = (299, 299)
    preprocess = preprocess_input


def img_and_lbl(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]

    return all_image_path, all_image_label


_, lbl = img_and_lbl(TRAIN_DIREC)


def data_generator():
    image_generator = ImageDataGenerator(validation_split=VAL_SIZE)

    train_ds_gen = image_generator.flow_from_directory(batch_size=32,
                                                       directory=TRAIN_DIREC,
                                                       shuffle=True,
                                                       target_size=img_size,
                                                       subset="training",
                                                       class_mode='categorical')

    val_ds_gen = image_generator.flow_from_directory(batch_size=32,
                                                     directory=TRAIN_DIREC,
                                                     shuffle=True,
                                                     target_size=img_size,
                                                     subset="validation",
                                                     class_mode='categorical')

    image_generator_test = ImageDataGenerator()
    test_ds_gen = image_generator_test.flow_from_directory(directory=TEST_DIREC,
                                                           shuffle=False,
                                                           target_size=img_size,
                                                           class_mode='categorical')

    return train_ds_gen, val_ds_gen, test_ds_gen


def train_model(args):
    # We set up the base model and train it using the 'imagenet' weights
    print("[INFO] loading {}...".format(args["model"]))
    Network = MODELS[args["model"]]
    base = Network(weights='imagenet',
                   input_shape=inputShape,
                   include_top=False)

    # We freeze the layers of the models and make them not trainable
    # before unfreezing and training them later on. We will first only
    # train on the top layer of the model
    base.trainable = False

    inputs = keras.Input(shape=inputShape)

    data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal"),
         layers.RandomRotation(0.2), ]
    )
    # We add a global spatial average pooling layer
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # We can try and add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    # And have a logistic layer
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # This is the model that will be trained
    model = keras.Model(inputs, outputs)

    # We compile the model (always do after setting layers to non-trainable)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=[CategoricalAccuracy()])

    # We use the ImageGenerator keras function to generate the training, and validation
    # sets from the original training 'img' folder
    train_im_gen = ImageDataGenerator(rotation_range=30,
                                      zoom_range=0.15,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    val_im_gen = ImageDataGenerator(validation_split=VAL_SIZE)

    train_ds_gen = train_im_gen.flow_from_directory(batch_size=32,
                                                    directory=TRAIN_DIREC,
                                                    seed=42,
                                                    shuffle=True,
                                                    target_size=img_size,
                                                    subset="training",
                                                    class_mode='categorical')

    val_ds_gen = val_im_gen.flow_from_directory(batch_size=32,
                                                directory=TRAIN_DIREC,
                                                seed=42,
                                                shuffle=True,
                                                target_size=img_size,
                                                subset="validation",
                                                class_mode='categorical')

    # As the data classes are imbalanced, we use the 'compute_class_weight' to balance them
    class_weights = class_weight.compute_class_weight("balanced", np.unique(lbl), [0, 1, 2, 3, 4, 5])
    train_class_weights = dict(enumerate(class_weights))
    print(train_class_weights)  # {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    print(model.summary())

    # We then use the 'fit' function to train the top layer of the model
    history = model.fit(x=train_ds_gen,
                        epochs=NUM_EPOCHS,
                        validation_data=val_ds_gen,
                        # class_weight=train_class_weights,
                        callbacks=[callback])

    # In order to fine tune the model and improve its performance
    # we can make this part of the code accessible and run it
    if False:
        base.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-6),
                      loss=CategoricalCrossentropy(from_logits=False),
                      metrics=[CategoricalAccuracy()])

        history = model.fit(x=train_ds_gen,
                            epochs=NUM_EPOCHS_FINE_TUNE,
                            validation_data=val_ds_gen,
                            class_weight=train_class_weights,
                            callbacks=[callback])

    for i, layer in enumerate(base.layers):
        print("[INFO] This is layer", i, layer.name, " in the base model.")

    # We then print the model summary
    print(model.summary())

    # summarize history for accuracy
    fig, ax = plt.subplots(2)
    fig.suptitle('{} model'.format(args['model']))
    ax[0].plot(history.history['categorical_accuracy'], color='r')
    ax[0].plot(history.history['val_categorical_accuracy'], color='g')
    # ax[0].set_title('model accuracy')
    ax[0].set(ylabel='accuracy')
    # ax[0].ylabel('accuracy')
    # ax[0].xlabel('epoch')
    ax[0].grid()
    # plt.title('History for training and validation accuracies for {} model'.format(args['model']))
    # plt.title('{} model'.format(args['model']))
    ax[0].legend(['train', 'validation'], loc='upper left')
    # plt.savefig(
    #     "C:/Users/Zohra/MScProject/classification/plots/Accuracy/" + 'Accuracy for {} training'.format(args['model']))
    # plt.show()

    # summarize history for loss
    ax[1].plot(history.history['loss'], color='k')
    ax[1].plot(history.history['val_loss'], color='b')
    # ax[1].set_title('model loss')
    ax[1].set(xlabel='epochs', ylabel='loss')
    # ax[1].ylabel('loss')
    # ax[1].xlabel('epoch')
    ax[1].grid()
    # plt.title('History for training and validation losses for {} model'.format(args['model']))
    # plt.title('{} model'.format(args['model']))
    ax[1].legend(['train', 'validation'], loc='upper left')
    plt.savefig("C:/Users/Zohra/MScProject/classification/plots/base_models_classweights_data_augmentation/Acc-Loss/" +
                'Acc-Loss for {} training'.format(args['model']))
    plt.show()

    return model


def save_model(model, folder=MODEL_FOLDER):
    print(f"saving model to {folder} ")
    model.save(folder)


def load_model(folder=MODEL_FOLDER):
    print(f"loading model from {folder} ")
    model = keras.models.load_model(folder)
    return model


def test_model(model):
    # We then use the ImageDataGenerator function to generate the test dataset
    # from the 'test' directory
    image_generator_test = ImageDataGenerator()
    test_ds_gen = image_generator_test.flow_from_directory(directory=TEST_DIREC,
                                                           shuffle=False,
                                                           target_size=img_size,
                                                           class_mode='categorical')

    test_labels = test_ds_gen.labels
    # The 'evaluate' function is then used to determine the loss score and
    # accuracy of the model
    score, acc = model.evaluate(x=test_ds_gen, verbose=1)
    print('[INFO] Loss Score: ', score)
    print('[INFO] Accuracy: ', acc)
    # print('[INFO] kmetric: ', kmetric)

    # The 'predict' function is then use to make predictions
    # by providing us with logits values, which we convert into label values
    predicts_logits = model.predict(test_ds_gen)
    predicts = [np.argmax(l) for l in predicts_logits]

    # We define the confusion matrix using sklearn and display it
    conf_matrix = confusion_matrix(test_labels, predicts, labels=[0, 1, 2, 3, 4, 5], normalize="all")
    conf_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["cardboard", "glass", "metal",
                                                                    "paper", "plastic", "trash"])
    # print(conf_matrix)
    conf_disp.plot()
    plt.savefig(
        "C:/Users/Zohra/MScProject/classification/plots/base_models_classweights_data_augmentation/ConfMatrix/" +
        'Confusion_Matrix for {}'.format(args['model']))
    # plt.show()

    report = classification_report(y_true=test_labels, y_pred=predicts, target_names=["cardboard", "glass", "metal",
                                                                                      "paper", "plastic", "trash"])
    # sns.heatmap(pd.DataFrame(report).iloc[:, :].T, annot=True)
    # plt.show()
    print('Classification Report for {} model:'.format(args['model']), report)

    return predicts == test_labels


def classify_image(model, img, fn_image):
    # resize the image
    resized_img = cv2.cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.imshow(resized_img)
    import os
    fp = os.path.join(
        r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\base_models_and_coco_labels\ssd"
        r"\resized_images",
        fn_image)
    plt.savefig(fp)

    img = np.expand_dims(resized_img, axis=0)
    # The 'predict' function is then use to make predictions
    # by providing us with logits values, which we convert into label values

    predicts_logits = model.predict(img)
    predicts = [np.argmax(l) for l in predicts_logits]

    return LABELS[predicts[0]]


def classify_detections(model):
    image_info = image_boxes_params.load_image_labels(FP_ANNOTATIONS)

    image_paths = get_local_paths(FOLDER_DETECTED_IMAGES)
    predicted = []
    actual = []
    predicted_actual_tuples = {}
    for fp in image_info:
        # print(fp)
        image_fn = os.path.basename(fp)
        # if fp in ['collage662.png', 'collage666.png', 'collage675.png', 'collage677.png', 'collage682.png', 'collage692.png', 'collage723.png','collage740.png', 'collage743.png', 'collage748.png']:
        if fp in ['collage633.png', 'collage725.png', 'collage748.png']:
            continue
        img = load_image_into_numpy_array(os.path.join(FOLDER_DETECTED_IMAGES, fp))
        # print('img', img)
        predicted_label = classify_image(model, img, image_fn)
        for label in image_info[fp]:
            # print('label', label)
            # if (predicted_label, label) not in predicted_actual_tuples:
            #     predicted_actual_tuples[(predicted_label, label)] = 1
            # else:
            #     predicted_actual_tuples[(predicted_label, label)] += 1
            actual.append(label)
            predicted.append(predicted_label)
        # print(predicted_label, target_label

    # print(actual)
    # print(predicted)
    conf_matrix = confusion_matrix(actual, predicted, labels=LABELS, normalize="all")
    conf_matrix = 100 * conf_matrix
    conf_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=LABELS)
    # print(conf_matrix)
    conf_disp.plot()
    plt.savefig(
        r"C:\Users\Zohra\MScProject\classification\plots\cropped_imgs"
        r"\ssd\experiment2\Confusion_Matrix_det_classifier_collage.jpg")


def train_and_test():
    model = train_model(args)
    save_model(model, MODEL_FOLDER)
    model = load_model(model)

    byitem_results = test_model(model)


def test_itembyitem_inference(models):
    cols = None
    for mf in models:
        model_folder = os.path.join(ROOT_FOLDER, mf)
        model = load_model(model_folder)
        byitem_results = test_model(model)
        # byitem_results = np.array(byitem_result)
        if cols is None:
            cols = byitem_results
        else:
            cols = np.vstack([cols, byitem_results])

    return cols


def t_test(correct1, correct2, n, model1, model2):
    p1 = correct1 / n
    p2 = correct2 / n
    p = (correct1 + correct2) / (2 * n)

    print(correct1, correct2, p1, p2, p, n)
    z = (p1 - p2) / (np.sqrt((2 * p * (1 - p)) / n))

    print(p1, p2, z)
    z05 = -1.645
    if z < z05:
        print(f"reject H0, {model2} is better than {model1}")
    else:
        print(f"cant rejcct H0,{model2} is NOT better than {model1}")



def stat_tests(table, model1, model2):
    # define contingency table
    # Efficient:  Accuracy:  0.837, Wrong : 0.173
    # calculate mcnemar test
    # result = mcnemar(table, exact=True)

    if table[0, 1] + table[1, 0] == 0:
        print(f'{model1} and {model2} have exactly same errors')
        return

    corrected = (table[0, 1] <= 25 or table[1, 0] >= 25)
    exact = not corrected
    chi2, p = mcnemar(ary=table, exact=exact, corrected=corrected)
    # summarize the finding
    # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    print('statistic=%.3f, p-value=%.3f' % (chi2, p))
    # interpret the p-value
    alpha = 0.05
    # if result.pvalue > alpha:
    if p > alpha:
        print(f'{model1} and {model2} have same proportions of errors (fail to reject H0) on test set')
    else:
        print(f'{model1} and {model2} have Different proportions of errors (reject H0) on the test')


if __name__ == "__main__":
    # train_and_test()
    # models = ["my_EfficientNet_model", "my_VGG16_model","my_InceptionV3_model", "my_Resnet50_model", "my_Resnet101_model", "my_Xception_model", "my_VGG19_model"]
    # cols = test_itembyitem_inference(models)
    model = load_model()
    # test_model(model)
    classify_detections(model)

    # n = cols.shape[1]
    #
    # for i in range(1, len(models)):
    #     correct1 = np.sum(cols[0, :] == True)
    #     correct2 = np.sum(cols[i, :] == True)
    #
    #     TT = np.sum(np.logical_and(cols[0, :] == True, cols[i, :] == True))
    #     TF = np.sum(np.logical_and(cols[0, :] == True, cols[i, :] == False))
    #     FT = np.sum(np.logical_and(cols[0, :] == False, cols[i, :] == True))
    #     FF = np.sum(np.logical_and(cols[0, :] == False, cols[i, :] == False))
    #     print(TT, TF, FT, FF)
    #
    #     table = np.array([[TT, TF], [FT, FF]])
    #     stat_tests(table, models[0], models[i])
    #     t_test(correct1, correct2, n, models[0], models[i])

#########################################
# Links:
# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# https://stackoverflow.com/questions/58655785/evaluate-keras-model-with-image-data-generator
# https://stackoverflow.com/questions/36177183/python-errno13-permission-denied
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_crossentropy
# https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
