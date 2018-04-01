#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

import database.data


def get_estimator():
    return tf.estimator.Estimator(
        model,
        model_dir='models/cnn_mnist_like',
        params={},
    )

def model(features, labels, mode, config, params):
    """
    Convolutional neural network model.
        features - batch of 3-channel images
        labels - batch of labels (single int)
    """
    ModeKeys = tf.estimator.ModeKeys
    images = features
    # images = tf.Print(images, [images], message="This is a: ")

    # convolutional part
    conv1 = tf.layers.conv2d(images, filters=32, kernel_size=5, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    # dense part
    flat = tf.layers.flatten(pool2)
    dense1 = tf.layers.dense(flat, 1024, activation=tf.nn.relu)
    dense1_dropout = tf.layers.dropout(dense1, rate=.4, training=mode == ModeKeys.TRAIN)
    logits = tf.layers.dense(dense1_dropout, database.data.N_CLASSES)

    if mode == ModeKeys.PREDICT:
        predictions = {
            'predictions': tf.argmax(logits, axis=1),  # index of best prediction for each image
            'logits': logits,
            'probabilities': tf.nn.softmax(logits),
            'top_indices': tf.nn.top_k(logits, k=database.data.N_CLASSES).indices,
        }
        return tf.estimator.EstimatorSpec(mode, predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimization = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimization)

    assert mode == ModeKeys.EVAL, 'Received unexpected mode: %s' % mode
    metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(logits, axis=1)),
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
