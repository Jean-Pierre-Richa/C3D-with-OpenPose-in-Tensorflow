import tensorflow as tf
import cv2
from tensorflow.python.client import timeline
from openpose.common import estimate_pose, draw_humans
from openpose.networks import get_network
import numpy as np
import os

cwd = os.getcwd()

model = 'mobilenet'
input_width = 368
input_height = 368
stage_level = 6

# tf.reset_default_graph()
input_node = tf.placeholder(tf.float32, shape=(1, input_height, input_width, 3), name = 'image')

first_time = True
net, _, last_layer = get_network(model, input_node, None)


def compute_pose_frame(input_image, sess):
    try:
        image = cv2.resize(input_image, dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
        # print(image)
        global first_time
        if first_time:
            #load pretrained weights
            ckpts = cwd + '/openpose/models/trained/mobilenet_368x368/model-release'

            vars_in_checkpoint = tf.train.list_variables(ckpts)
            # print('vars_in_checkpoint: ', vars_in_checkpoint)
            var_rest = []
            for el in vars_in_checkpoint:
                var_rest.append(el[0])
            # print('var_rest: ', var_rest)
            variables = tf.contrib.slim.get_variables_to_restore()
            # print('variables: ', variables)
            var_list = [v for v in variables if v.name.split(':')[0] in var_rest]

            loader = tf.train.Saver(var_list=var_list)
            loader.restore(sess, ckpts)

            first_time = False
    except (RuntimeError):
        print("Error, couldn't generate pose")
        pass


    vec = sess.run(net.get_output(name = 'concat_stage7'), feed_dict={'image:0': [image]})
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    pafMat, heatMat = sess.run(
        [
            net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
            net.get_output(name=last_layer.format(stage=stage_level, aux=2))
        ], feed_dict={'image:0': [image]}
    )
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    heatMat, pafMat = heatMat[0], pafMat[0]
    humans = estimate_pose(heatMat, pafMat)
    pose_image = np.zeros(tuple(image.shape), dtype=np.uint8)
    pose_image = draw_humans(pose_image, humans)

    # cv2.imwrite('test.png', pose_image)

    return pose_image
