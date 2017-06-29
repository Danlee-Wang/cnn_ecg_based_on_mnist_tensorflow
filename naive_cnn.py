import numpy as np
import tensorflow as tf

import cnn_ecg_input

SEG_LENGTH = 512
batch_size = 128

train_data, train_labels, eval_data, eval_labels, test_data, test_labels = cnn_ecg_input.inputs('afdb_data/')


ecg_placeholder = tf.placeholder(tf.float32, shape=(batch_size, SEG_LENGTH))
lab_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

with tf.name_scope('conv1') as scope:
