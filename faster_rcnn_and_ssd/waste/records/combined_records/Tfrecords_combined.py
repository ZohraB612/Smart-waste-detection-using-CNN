import os
import tensorflow as tf

path_to_records = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records"
path_train_records = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records\training_records"
path_val_records = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records\validation_records"

five_per = []
twenty_five_per = []
fifty_per = []
hundred_per = []

# for file in os.listdir(path_train_records):
for file in os.listdir(path_val_records):
    if "0.05" in file:
        # five_per.append(path_train_records + "/" + file)
        five_per.append(path_val_records + "/" + file)
    elif "0.25" in file:
        # twenty_five_per.append(path_train_records + "/" + file)
        twenty_five_per.append(path_val_records + "/" + file)

    elif "0.5" in file:
        # fifty_per.append(path_train_records + "/" + file)
        fifty_per.append(path_val_records + "/" + file)

    elif "1" in file:
        # hundred_per.append(path_train_records + "/" + file)
        hundred_per.append(path_val_records + "/" + file)

    else:
        continue

# filenames = {"five" : five_per, "twenty-five" : twenty_five_per, "fifty" : fifty_per, "hundred" :hundred_per}
filenames = [five_per, twenty_five_per, fifty_per, hundred_per]
names = ['five_per', 'twenty_five_per', 'fifty_per', 'hundred_per']

outputPath = path_to_records + r"\combined_records"
# dataset = {}

i = 0
for filename in filenames:
    print(filename)
    # name = f'training{names[i]}.record'
    name = f'validation{names[i]}.record'

    # writer = tf.io.TFRecordWriter(outputPath + "/" + filename)
    dataset = tf.data.TFRecordDataset(filename)
    # print(dataset[filename])
    writer = tf.data.experimental.TFRecordWriter(name)
    writer.write(dataset)
    i += 1

# Links
# https://stackoverflow.com/questions/3437059/does-python-have-a-string-contains-substring-method
