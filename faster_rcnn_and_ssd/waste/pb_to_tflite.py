import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\tflite_export_raspberry_pi\saved_model')
tflite_model = converter.convert()
open('converted_model.tflite', 'wb').write(tflite_model)