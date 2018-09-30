import os
import cv2
import json
import c3d_model
import numpy as np
import tensorflow as tf
import activities
import PoseEstimation
from tqdm import tqdm
import argparse

json_path = 'json/'
videos_path = 'videos/'
test = 'test'
train = 'train'

frames_per_step = activities.frames_per_step
im_size = activities.im_size

def generateFiles(json_path):

    training_list = []
    testing_list = []

    for f in os.listdir(json_path):
        with open(json_path + f) as file:
            json_dict = json.load(file)
            for vid_name, primitives in json_dict.items():
                for primitive in primitives:
                    if primitive['label'] in activities.activities_tfrecords:
                        segment = primitive['milliseconds']
                        path = videos_path + vid_name
                        for j, k in activities.activities_tfrecords.items():
                            if j == primitive['label']:
                                id = k
                        if 'train' in f:
                            training_list.append([path, segment, primitive['label'], id])
                        else:
                            testing_list.append([path, segment, primitive['label'], id])
    return training_list, testing_list

def generate_tfRecords(phase, file_list):

    train_filename = 'tfrecords/' + phase + '.tfrecords'  # address to save the TFRecords file

    # Create a session for running Ops on the Graph and initialize the variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # open the TFRecords file
    with tf.python_io.TFRecordWriter(train_filename) as writer:
        for i in tqdm(range(len(file_list))):
            iter = file_list[i]
            # Load the image
            frames = get_frames(iter[0], frames_per_step, iter[1], im_size, sess)
            label = iter[3]
            # Create the features
            feature = {}
            feature['class_label'] = _int64_feature(label)
            for j in range(len(frames)):
                ret, buffer = cv2.imencode(".jpg", frames[j])
                feature["frames/{:02d}".format(j)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
             # Serialize to string and write on the file
            writer.write(example.SerializeToString())
    sys.stdout.flush()

def get_frames(path, frames_per_step, segment, im_size, sess):

    video = cv2.VideoCapture(path)
    fps = (video.get(cv2.CAP_PROP_FPS))
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # check segment consistency
    if (max_len < segment[1]):
        segment[1] = max_len

    # define start frame
    central_frame = (np.linspace(segment[0], segment[1], num=3)) / 1000 * fps
    start_frame = central_frame[1] - frames_per_step / 2

    # for every frame in the clip extract frame, compute pose and insert result
    # in the matrix
    frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)
    for z in range(frames_per_step):
        frame = start_frame + z
        video.set(1, frame)
        ret, im = video.read()
        pose_frame = PoseEstimation.compute_pose_frame(im, sess)
        res = cv2.resize(pose_frame, dsize = (im_size, im_size),
                         interpolation=cv2.INTER_CUBIC)
        frames[z, :, :, :] = res
    return frames
# Convert data to features
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(json, videos):
    training_list, testing_list = generateFiles(json)
    generate_tfRecords(test, testing_list)
    generate_tfRecords(train, training_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tfrecords')
    parser.add_argument('--json', dest='json', type=str, default='json/', help='path of the json files')
    parser.add_argument('--videos', dest='videos', type=str, default='videos/', help='path of the video files')
    args = parser.parse_args()

    main(args.json, args.videos)
