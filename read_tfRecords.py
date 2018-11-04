import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import activities

frames_per_step = activities.frames_per_step
path='tfrecords/'
def extract_tfRecords(Batch_size, phase, sess):

    data_path = path + phase + '.tfrecords'
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Create a list of filenames and pass it to a queue
    # Define a reader and read the next record
    # Decode the record read by the reader
    filename_queue=tf.train.string_input_producer([data_path])
    reader=tf.TFRecordReader()
    _, serialized_example=reader.read(filename_queue)


    feature = dict()
    feature["class_label"] = tf.FixedLenFeature((), tf.int64)
    for i in range(activities.frames_per_step):
        feature["frames/{:02d}".format(i)] = tf.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.parse_single_example(serialized_example, feature)

    # Decode the image
    image = []
    for i in range(activities.frames_per_step):
        image.append(tf.image.decode_jpeg(parsed_features["frames/{:02d}".format(i)]))

    # put the frames into a big tensor of shape (N,H,W,3)
    image = tf.stack(image)
    label = tf.cast(parsed_features['class_label'], tf.int64)

    # Reshape image into the original shape
    image = tf.reshape(image, [frames_per_step, activities.im_size, activities.im_size, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=Batch_size,
                            capacity=1000, num_threads=1, min_after_dequeue=100)

    # Initialize variables
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for batch_index in range(Batch_size):
        train_images, train_labels = sess.run([images, labels])
        train_images = train_images.astype(np.uint8)
    return train_images, train_labels
        # for j in range(6):
        #     plt.subplot(2, 3, j+1)
        #     plt.imshow(train_images[j, ...])
        #     for i, k in activities.activities_tfrecords.items():
        #         if train_labels[j]==k:
        #             plt.title(i)
        #
        # plt.show()

    # Stop the threads
    coord.request_stop()

    # wait for threads to stop
    coord.join(threads)
    sess.close
