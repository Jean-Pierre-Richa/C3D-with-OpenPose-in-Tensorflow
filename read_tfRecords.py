import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import activities

path='tfrecords/'
def extract_tfRecords(Batch_size, phase, sess):

    data_path = path + phase + '.tfrecords'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        feature={'image': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.int64)}

        # Create a list of filenames and pass it to a queue
        # Define a reader and read the next record
        # Decode the record read by the reader
        filename_queue=tf.train.string_input_producer([data_path])
        reader=tf.TFRecordReader()
        _, serialized_example=reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image from string back to numbers
        # Cast label data into int32
        # Reshape image into the original shape
        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.cast(features['label'], tf.int32)
        image = tf.reshape(image, [activities.im_size, activities.im_size, 3])

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([image, label], batch_size=Batch_size,
                                capacity=30, num_threads=1, min_after_dequeue=10)

        # Initialize variables
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_index in range(5):
            img, lbl = sess.run([images, labels])
            img = img.astype(np.uint8)

            for j in range(6):
                plt.subplot(2, 3, j+1)
                plt.imshow(img[j, ...])
                for i, k in activities.activities_tfrecords.items():
                    if lbl[j]==k:
                        plt.title(i)

            plt.show()

        # Stop the threads
        coord.request_stop()

        # wait for threads to stop
        coord.join(threads)
        sess.close