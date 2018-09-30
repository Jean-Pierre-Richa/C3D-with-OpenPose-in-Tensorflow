# from random import shuffle
import glob
import os
import activities
import tensorflow as tf
import sys
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

shuffle_data = True

DATASET_DIR_TRAIN = 'images/train/'
DATASET_DIR_TEST = 'images/test/'
TFR_DIR = 'tfrecords/'
# Read addresses and labels from the 'train' folder
def create_tfRecords(DATASET_DIR, phase):
    labels = []
    labels = activities.activities_tfrecords
    train_filename = TFR_DIR + phase + '.tfrecords'
    if not os.path.exists(TFR_DIR):
        os.makedirs(TFR_DIR)

    if phase == 'train':
        # addrs = glob.glob(DATASET_DIR_TRAIN)
        addrs = DATASET_DIR_TRAIN
    else:
        # addrs = glob.glob(DATASET_DIR_TEST)
        addrs = DATASET_DIR_TEST
    # phase_addrs = phase + '_addrs'
    # phase_labels = phase + '_labels'
    phase_addrs = []
    phase_labels = []

    for classes in os.listdir(DATASET_DIR):
        final_class = classes
        for j, k in labels.items():
            if final_class.split('.')[0] in j:
                phase_addrs.append(final_class)
                phase_labels.append(k)
                # print(final_class)
    # print(phase_addrs)
    # print(phase_labels)
    # Open the tfrecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    for i in tqdm(range(len(phase_addrs))):
        # print how many images are saved every 1000 images
        # if not i%1000:
        #     print(phase + ' data: {}/{}'.format(i, len(phase_addrs)))
        #     sys.stdout.flush()

        img = load_image(addrs+phase_addrs[i])
        lbl = phase_labels[i]
        print(addrs+phase_addrs[i])
        print(lbl)

            # print ('label: ', lbl)
            # create a feature
        feature = {'label': _int64_feature(lbl),
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

            # create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    # img1=mpimg.imread(addrs+phase_addrs[i])
    # imgplot = plt.imshow(img1)
    # plt.show()
    writer.close()
    sys.stdout.flush()

def load_image(addr):
    # Read an image and resize to (im_size, im_size)
    # cv2 load image as BGR, convert it to RGB
    # print (addr)
    img = cv2.imread(addr)
    img = cv2.resize(img, (activities.im_size, activities.im_size), interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # print (type(img))
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


create_tfRecords('images/train/', 'train')
