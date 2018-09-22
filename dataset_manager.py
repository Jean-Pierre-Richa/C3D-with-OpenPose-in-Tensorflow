import json
import os
import cv2
import numpy as np
import random
import pprint
import sys
from numpy import *
from os.path import dirname, realpath
dir_path = dirname(realpath(__file__))
print('dir_path: ' + dir_path)
project_path = realpath(dir_path + '/..')
print('project_path' + project_path)
libs_dir_path = dir_path+ '/openpose'
print('libs_dir_path' + libs_dir_path)
sys.path.append('libs_dir_path' + libs_dir_path)
import PoseEstimation

def get_frames(video_path, frames_per_step, segment, im_size, sess):
    #load video and acquire its parameters usingopencv
    # video_path = '/H3.6M/Directions/S5_Directions 2.55011271.mp4'
    video = cv2.VideoCapture(video_path)
    fps = (video.get(cv2.CAP_PROP_FPS))
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # check segment consistency
    if (max_len < segment[1]):
        segment[1] = max_len


    #define start frame

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

def read_clip_and_label(Batch_size, trainin, frames_per_step, im_size, sess):
    batch = np.zeros(shape=(Batch_size, frames_per_step, im_size, im_size, 3), dtype=float)
    labels = np.zeros(shape=(Batch_size), dtype=int)
    # print ('labels: ', labels)
    # print ('Training: ', trainin)
    for s in range(Batch_size):
        if trainin == 1:
            with open('dataset_training.json') as file:
                Json_dict = json.load(file)
                entry_to_path = 'H3.6M/train/'
                entry_name = random.choice(list(Json_dict.keys()))
                # print('entry name: ', entry_name)
                training_entry = random.choice(Json_dict[entry_name])
                # print('training entry:', training_entry)
                path = entry_to_path + entry_name

                labels_list = []
                c = 0

                # Append the labels 1 time to have the classes labels
                for label in Json_dict[entry_name]:
                    # print(label)
                    if (label['label'] not in labels_list):
                        labels_list.append(label['label'])
                        c = c+1
                # print(c, 'labels_list')
                # id_to_label = dict(enumerate(labels))
                # print ('ID to labels: ', id_to_label )
        elif trainin == 2:
            with open('dataset_test.json') as file:
                Json_dict = json.load(file)
                entry_to_path = 'H3.6M/test/'
                entry_name = random.choice(list(Json_dict.keys()))
                # print('entry name: ', entry_name)
                training_entry = random.choice(Json_dict[entry_name])
                # print('training entry:', training_entry)
                path = entry_to_path + entry_name

                labels_list = []
                c = 0

                # Append the labels 1 time to have the classes labels
                for label in Json_dict[entry_name]:
                    # print(label)
                    if (label['label'] not in labels_list):
                        labels_list.append(label['label'])
                        c = c+1
                # print(c, 'labels_list')
                # id_to_label = dict(enumerate(labels))
                # print ('ID to labels: ', id_to_label )
        label_to_id = dict(map(reversed, enumerate(labels_list)))
        # print('label to ID: ', label_to_id)

        segment = training_entry['milliseconds']

        clip = get_frames(path, frames_per_step, segment, im_size, sess)
        batch[s, :, :, :, :] = clip
        labels[s] = label_to_id[training_entry['label']]
        # print('labels[s]: ', labels[s])
        # print ('s: ', s)
        # print('labels: ', labels)
    return batch, labels
