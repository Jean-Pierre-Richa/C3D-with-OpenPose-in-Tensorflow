import os
import cv2
import json
import c3d_model
import numpy as np
import skvideo.io
import tensorflow as tf
import activities
import PoseEstimation
import tqdm
json_path = 'json/'
test_json = 'dataset_testing.json'
train_json = 'dataset_training.json'

frames_per_step = c3d_model.NUM_FRAMES_PER_CLIP
im_size=c3d_model.CROP_SIZE

videos_path = 'videos/'

test='test/'
train='train/'

directory = './images/'

def generateDataset(file_json, kind):
    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(json_path + file_json) as file:
        json_dict = json.load(file)
        for vid_name, primitives in json_dict.items():
            print(vid_name)
            for primitive in primitives:
                segment = primitive['milliseconds']
                segment1 = segment[0]
                segment2 = segment[1]
                for key, label in primitive.items():
                    if key == 'label':
                        print('label: ', label)
                        for i, j in activities.activities_tfrecords.items():
                            DIRECTORY= directory + kind + str(label)
                            if not os.path.exists(DIRECTORY):
                                os.makedirs(DIRECTORY)
                            get_frames(DIRECTORY, vid_name, videos_path, label, activities.frames_per_step, segment2, segment1, activities.im_size, sess)
                            break
def get_frames(DIRECTORY, vid_name, video_path, label, frames_per_step, segment2, segment1, im_size, sess):
    # load video and acquire its parameters usingopencv
    # video_path = '/H3.6M/Directions/S5_Directions 2.55011271.mp4'
    video = cv2.VideoCapture(videos_path+vid_name)
    fps = (video.get(cv2.CAP_PROP_FPS))
    # video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # check segment consistency
    # if (max_len < segment2[0]):
    #     segment2[0] = max_len

    #define start frame

    central_frame = (np.linspace(segment1, segment2, num=3)) / 1000 * fps
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
        cv2.imwrite(DIRECTORY + '/' + str(label) + '.' + '[' + str(segment1) + ',' + str(segment2) + ']' + '.' + str(z) + '.jpg', res)
    print('saved: ' + str(label) + '.' + '[' + str(segment1) + ',' + str(segment2) + ']')
    return frames
generateDataset(file_json=test_json, kind=test)
# generateDataset(file_json=train_json, kind=train)
