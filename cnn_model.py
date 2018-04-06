#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

import data


def get_estimator():
    return tf.estimator.Estimator(
        model,
        model_dir='models/cnn_v2_1',
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

    # convolutional part (dimensions without first (batch_size))
    net = images  # [50, 50, 1]
    net = tf.layers.conv2d(images, filters=48, kernel_size=7, activation=tf.nn.relu)  # [44, 44, 48]
    net = tf.layers.max_pooling2d(net, 2, 2)  # [22, 22, 48]
    net = tf.layers.conv2d(net, filters=64, kernel_size=5, activation=tf.nn.relu)  # [18, 18, 64]
    # 1x1 convolution
    net = tf.layers.conv2d(net, filters=16, kernel_size=1, activation=tf.nn.relu)  # [18, 18, 16]
    net = tf.layers.max_pooling2d(net, 3, 3)  # [6, 6, 16]
    net = tf.layers.conv2d(net, filters=64, kernel_size=3, activation=tf.nn.relu)  # [4, 4, 64]
    # dense part
    net = tf.layers.flatten(net)  # [1024]
    net = tf.layers.dropout(net, rate=.7, training=mode == ModeKeys.TRAIN)
    net = tf.layers.dense(net, 768, activation=tf.nn.relu)  # [768]
    net = tf.layers.dropout(net, rate=.7, training=mode == ModeKeys.TRAIN)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        optimization = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimization)

    assert mode == ModeKeys.EVAL, 'Received unexpected mode: %s' % mode
    metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(logits, axis=1)),
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
