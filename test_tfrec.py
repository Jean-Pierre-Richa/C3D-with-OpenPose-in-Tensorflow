import tensorflow as tf

for example in tf.python_io.tf_record_iterator("tfrecords/train.tfrecords"):
  result = tf.train.Example.FromString(example)
  print(result.features.feature['label'].int64_list.value)
  print(result.features.feature['image'].bytes_list.value)
