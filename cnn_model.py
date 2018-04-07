#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

import data


def get_estimator():
    return tf.estimator.Estimator(
        model,
        model_dir='models/cnn_v8_38x38-4',
        params={},
    )

def model(features, labels, mode, config, params):
    """
    Convolutional neural network model.
        features - batch of 3-channel images, float32 (0-255)
        labels - batch of labels (single int)
    """
    ModeKeys = tf.estimator.ModeKeys
    images = features

    tf.assert_equal(tf.shape(images)[1:3], data.Database.IMAGE_SIZE,
        data=[tf.shape(images)], summarize=5, message='Wrong input images shape')

    # convolutional part (dimensions without first (batch_size))
    net = images  # [38, 38, 1]
    net = tf.layers.conv2d(images, filters=8, kernel_size=3, activation=tf.nn.relu)  # [36, 36, 8]
    net = tf.layers.max_pooling2d(net, 2, 2)  # [18, 18, 8]
    net = tf.layers.conv2d(net, filters=32, kernel_size=3, activation=tf.nn.relu)  # [16, 16, 32]
    net = tf.layers.max_pooling2d(net, 2, 2)  # [8, 8, 32]
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, activation=tf.nn.relu)  # [6, 6, 128]
    net = tf.layers.max_pooling2d(net, 2, 2)  # [3, 3, 128]
    net = tf.layers.conv2d(net, filters=512, kernel_size=3, activation=tf.nn.relu)  # [1, 1, 512]

    # dense part
    net = tf.layers.flatten(net)  # [512]
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)  # [1024]
    net = tf.layers.dropout(net, rate=.8, training=mode == ModeKeys.TRAIN)
    net = tf.layers.dense(net, data.Database.N_CLASSES)
    logits = net

    probabilities = tf.nn.softmax(logits)

    if mode == ModeKeys.PREDICT:
        predictions = {
            'predictions': tf.argmax(logits, axis=1),  # index of best prediction for each image
            'logits': logits,
            'probabilities': probabilities,
            'top_indices': tf.nn.top_k(logits, k=data.Database.N_CLASSES).indices,
        }
        return tf.estimator.EstimatorSpec(mode, predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        optimization = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimization)

    assert mode == ModeKeys.EVAL, 'Received unexpected mode: %s' % mode
    metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(logits, axis=1)),
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
