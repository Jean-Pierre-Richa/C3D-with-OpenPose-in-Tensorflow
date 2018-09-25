import json
import os
import cv2
import numpy as np
import random
import pprint
import sys
from numpy import *
import activities
from os.path import dirname, realpath
dir_path = dirname(realpath(__file__))
print('dir_path: ' + dir_path)
project_path = realpath(dir_path + '/..')
print('project_path' + project_path)
libs_dir_path = dir_path+ '/openpose'
print('libs_dir_path' + libs_dir_path)
sys.path.append('libs_dir_path' + libs_dir_path)
import PoseEstimation
json_path = 'json/'
entry_to_path = 'H3.6M/'
activities_ids = activities.activities_ids
def get_frames(video_path, frames_per_step, segment, im_size, sess):
    # load video and acquire its parameters usingopencv
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

def read_clip_and_label(Batch_size, frames_per_step, im_size, sess, test=False):
    batch = np.zeros(shape=(Batch_size, frames_per_step, im_size, im_size, 3), dtype=float)
    labels = np.zeros(shape=(Batch_size), dtype=int)
    # print ('labels: ', labels)
    # print ('Training: ', trainin)
    if test:
        dataset = 'dataset_testing.json'
    else:
        dataset = 'dataset_training.json'

    for s in range(Batch_size):
        with open(json_path + dataset) as file:
            Json_dict = json.load(file)
            entry_name = random.choice(list(Json_dict.keys()))
            # print('entry name: ', entry_name)
            training_entry = random.choice(Json_dict[entry_name])
            # print('training entry:', training_entry)
            path = entry_to_path + entry_name

        segment = training_entry['milliseconds']

        clip = get_frames(path, frames_per_step, segment, im_size, sess)
        batch[s, :, :, :, :] = clip
        labels[s] = activities_ids[training_entry['label']]
        # print('labels[s]: ', labels[s])
        # print ('s: ', s)
        # print('labels: ', labels)
    return batch, labels
