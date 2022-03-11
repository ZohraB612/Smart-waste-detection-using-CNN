from tensorflow.python.tools import freeze_graph
import os
import tensorflow as tf

freeze_graph.freeze_graph(r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records\classes.pbtxt', "", False,
                          r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\training\Faster_RCNN\experiment6\ckpt'
                          r'-11.index',
                          r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\outputs\new_output",
                          "save/restore_all",
                          "save/Const:0",
                          'frozentensorflowModel.pb', True, ""
                          )
