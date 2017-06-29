#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""


import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

import cnn_ecg_input

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  # input_layer = tf.reshape(features, [-1])
  input_layer = features

  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=32,
      kernel_size=10,
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2)

  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=64,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=2)

  # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  dense = tf.layers.dense(inputs=pool2, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=3)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # Load training and eval data
    # mnist = learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data, train_labels, eval_data, eval_labels, test_data, test_labels = cnn_ecg_input.inputs('afdb_data/')
    train_data = np.float32(train_data)
    eval_data = np.float32(eval_data)
    # Create the Estimator
    mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/cnn_ecg_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
