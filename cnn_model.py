#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

import log
import data


ModeKeys = tf.estimator.ModeKeys  # shorter


class Model:
    """
    Convolutional neural network model.
    Class defines computation graph and provides model_fn for tf.estimator.Estimator.
    """
    def __init__(self):
        self.logger = log.getLogger('model')
        self.model_dir = 'models/cnn_v9_94x94'
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
        self.layers = [
            # Convolutional part
            tf.layers.Conv2D(filters=8, kernel_size=3, activation=tf.nn.relu),   # [92, 92, 8]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [46, 46, 8]
            tf.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu),  # [44, 44, 8]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [22, 22, 8]
            tf.layers.Conv2D(filters=72, kernel_size=3, activation=tf.nn.relu),  # [20, 20, 32]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [10, 10, 32]
            tf.layers.Conv2D(filters=160, kernel_size=3, activation=tf.nn.relu), # [8, 8, 128]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [4, 4, 128]
            tf.layers.Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu), # [2, 2, 512]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [1, 1, 512]
            # Dense part
            tf.layers.Flatten(),                                                 # [512]
            tf.layers.Dense(units=1024, activation=tf.nn.relu),                  # [1024]
            tf.layers.Dropout(rate=.8),
            tf.layers.Dense(units=data.Database.N_CLASSES),
        ]
        self.intermediate_outputs = []

    def get_estimator(self, **kwargs):
        """Creates an instance of tf.estimator.Estimator for the model.

        If given, passes all keyword arguments to its constructor, else
        it uses model's default values."""
        return tf.estimator.Estimator(
            self.model_fn,
            model_dir=kwargs.get('model_dir', self.model_dir),
            **kwargs,
        )

    def model_fn(self, features, labels, mode, config=None, params={}):
        # features - batch of 1-channel images, float32, 0-255
        images = features
        assert images.shape[1:3] == data.Database.IMAGE_SIZE, \
            'Something is not yes! Wrong images.shape = %s' % images.shape

        # assemble model output from all layers
        self.logger.info('Building model...')
        net = images
        self.intermediate_outputs = []
        self.logger.info('   %s' % net.shape)
        for layer in self.layers:
            # for dropout we have to specify if it is training mode
            if isinstance(layer, tf.layers.Dropout):
                net = layer(net, training=mode == ModeKeys.TRAIN)
            else:
                net = layer(net)
            # save outputs of each layer
            self.intermediate_outputs.append(net)
            self.logger.info('   %s' % net.shape)

        # outputs (loss is computed if not in predict mode)
        logits = net
        probabilities = tf.nn.softmax(logits)
        loss_fn = lambda : tf.losses.sparse_softmax_cross_entropy(labels, logits)

        # create EstimatorSpecs depending on mode
        if mode == ModeKeys.PREDICT:
            predictions = {
                'predictions': tf.argmax(logits, axis=1),  # index of best prediction for each image
                'logits': logits,
                'probabilities': probabilities,
                'top_indices': tf.nn.top_k(logits, k=data.Database.N_CLASSES).indices,
            }
            if params.get('store_images', False):
                predictions['images'] = images
            if params.get('store_intermediate', False):
                intermediate_dict = {i: out for i, out in enumerate(self.intermediate_outputs)}
                predictions.update(intermediate_dict)
            return tf.estimator.EstimatorSpec(mode, predictions)

        loss = loss_fn()

        if mode == ModeKeys.TRAIN:
            optimization = self.optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimization)

        assert mode == ModeKeys.EVAL, 'Received unexpected mode: %s' % mode
        metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(logits, axis=1)), }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    def show_layers_outputs(self, predict_input_fn):
        """Shows outputs of intermediate layers according to specified filters

        Format of 'filter_string':
            <class_name> <nth_layer_of_that_type> <ith_filter> ...
        examples:
            1st conv2d, 3rd filter; 2nd conv2d, 3rd filter; 2nd max_pool2d, 3rd filter
            'Conv2D 1 3 Conv2D 2 3 MaxPooling2D 2 3'
            conv2ds from 2 to 4, filters from 1 to 9
            'Conv2D 2:4 1:9'
        """
        # convenience function
        def show_image(name, img):
            cv2.imshow(name, img)
            return not cv2.waitKey(0) in [27, ord('q')] # 'ESCAPE' or 'q'

        def best_grid(N):
            rows = np.floor(np.sqrt(N)) # floor => take less rows than columns
            cols = np.ceil(N / rows)    # take so many cols to fit all elements
            return int(rows), int(cols)

        # as for now only for 1 image
        params = {'store_intermediate': True, 'store_images': True}
        predictions = self.get_estimator(params=params).predict(predict_input_fn)

        # CNN-like layers
        for prediction in predictions:
            self.logger.info('Showing input image')
            show_image('image', prediction['images'])
            for i, layer in enumerate(self.layers):
                if isinstance(layer, tf.layers.Conv2D):
                    name = layer.__class__.__name__
                    filters = prediction[i]
                    n_filters = filters.shape[-1]
                    self.logger.info('Showing layer %s nr %d - %d filters' % (name, i+1, n_filters))
                    # create one big image from filter outputs
                    padding = 2
                    n_rows, n_cols = best_grid(n_filters)
                    im_height, im_width = filters.shape[:2]
                    image = 0.3 * np.ones([n_rows * im_height + (n_rows+1) * padding,
                        n_cols * im_width + (n_cols+1) * padding])
                    for row in range(n_rows):
                        for col in range(n_cols):
                            n = row * n_cols + col
                            if n >= n_filters:
                                continue
                            # find the right index ranges in the 'image' matrix
                            base_height = (row+1) * padding + row * im_height
                            base_width = (col+1) * padding + col * im_width
                            slice_h = slice(base_height, base_height + im_height)
                            slice_w = slice(base_width, base_width + im_width)
                            # normalize filter values to see activations
                            filter = filters[:, :, n]
                            eps = 1e-6
                            filter = filter / (np.max(filter) + eps) * 255
                            # insert filter image at its position
                            image[slice_h, slice_w] = filter
                    # increase image size to be visible
                    desired_rows = 500
                    scaling = desired_rows / image.shape[0]
                    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
                    to_continue = show_image('image', image)
                    if not to_continue:
                        break
