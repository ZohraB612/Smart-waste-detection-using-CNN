# Import libraries and packages
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.utils.np_utils import to_categorical
from numpy import argmax

from classify import data_generator
from keras.layers import Dense, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

np_config.enable_numpy_behavior()

MODEL_FOLDER = r"C:\Users\Zohra\MScProject\classification\models"
filename = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\annotations\wasteAnnotations1.csv'
path = r'C:\Users\Zohra\MScProject\classification'


def load_all_models():
    all_models = []
    model_names = ['my_Resnet50_model', 'my_VGG16_model',
                   'my_VGG19_model']  # , 'my_InceptionV3_model','my_Xception_model']
    for model_name in model_names:
        f_name = os.path.join(path, model_name)
        mdl = tf.keras.models.load_model(f_name)
        all_models.append(mdl)
        print('loaded:', f_name)
    return all_models


def save_model(model, folder=MODEL_FOLDER):
    print(f"saving model to {folder} ")
    model.save(folder)


models = load_all_models()
for i, model in enumerate(models):
    for layer in model.layers:
        layer.trainable = False

i = 0
for model in models:
    for layer in model.layers:
        layer._name = layer.name + str(i)
    i += 1

train_ds_gen, val_ds_gen, test_ds_gen = data_generator()
X_train, y_train = next(train_ds_gen)
X_test, y_test = next(val_ds_gen)


# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(3, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


ensemble_model = define_stacked_model(models)

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(X, inputy_enc, epochs=300, verbose=0)


fit_stacked_model(ensemble_model, X_test, y_test)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

hi = predict_stacked_model(ensemble_model, X_test)
hi = argmax(hi, axis=1)
accuracy = accuracy_score(y_test, hi)
print('Ensemble model Test Accuracy: %.3f' % accuracy)





# test_labels = test_ds_gen.labels
# # The 'evaluate' function is then used to determine the loss score and
# # accuracy of the model
# score, acc = model.evaluate(x=test_ds_gen, verbose=1)
# print('[INFO] Loss Score: ', score)
# print('[INFO] Accuracy: ', acc)
#
# # The 'predict' function is then use to make predictions
# # by providing us with logits values, which we convert into label values
# predicts_logits = model.predict(test_ds_gen)
# predicts = [np.argmax(l) for l in predicts_logits]
#
# # We define the confusion matrix using sklearn and display it
# conf_matrix = confusion_matrix(test_labels, predicts, labels=[0, 1, 2, 3, 4, 5], normalize="all")
# conf_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["cardboard", "glass", "metal",
#                                                                 "paper", "plastic", "trash"])

# def ensemble_predictions(models, testX):
#     yhats = [model.predict(testX) for model in models]
#     yhats = np.array(yhats)
#     # sum across ensemble members
#     summed = np.sum(yhats, axis=0)
#     # argmax across classes
#     result = np.argmax(summed, axis=1)
#     return result


# testing_dir = r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\images\obj_imgs\all_test'
# class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#
# res = load_image_boxes(filename)
# # print('this is res', res)
#
# # example of predicting
# correct_preds = 0
# incorrect_preds = 0
# total_preds = 0
#
# for file in os.listdir(testing_dir):
#     img = plt.imread(testing_dir + '/' + file)
#     # Resize it to the net input size:
#     img = cv2.resize(img, (224, 224))
#     img = tf.expand_dims(img, axis=0)
#
#     # Convert the data to float:
#     img = img.astype(np.float32)
#
#     class_index = ensemble_predictions(models, img)[0]
#
#     save_model(class_index, MODEL_FOLDER)
#     orig_label = res[file][4]
#     pred_label = class_names[class_index]
#     print(f'original label: {orig_label}, predicted label: {pred_label}')
#
#     if orig_label == pred_label:
#         correct_preds += 1
#     else:
#         incorrect_preds += 1
#
#     total_preds += 1
# accuracy = (correct_preds / total_preds) * 100
#
# print(f'Correct predictions: {correct_preds}, Incorrect predictions: {incorrect_preds}, '
#       f'Total predictions: {total_preds}')
# print(f'The ensemble model accuracy is: {accuracy}')

# Links
# https://github.com/hoanhle/Vehicle-Type-Detection/blob/master/test_trained_CNN.py
# https://towardsdatascience.com/destroy-image-classification-by-ensemble-of-pre-trained-models-f287513b7687
