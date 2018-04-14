#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import scipy.signal
import numpy as np
import tensorflow as tf

import log
import data
import cv2_show


ModeKeys = tf.estimator.ModeKeys  # shorter


# def cv2_show.show_image(image, wait=True):
#     """Shows given image using OpenCV.
#
#     'image' - image with float values from 0 to 255 (as Model operates on such)
#     'wait' - if True waits forever, if False waits 1ms, else waits 'wait' ms
#     return - True if ESCAPE or 'q' key was pressed else False
#     """
#     image = np.asarray(image)
#     assert image.dtype in [np.float32, np.float64], 'Image is not floating point'
#     cv2.imshow('image', image / 255)  # openCV wants floats in [0, 1]
#     wait_ms = (0 if wait else 1) if isinstance(wait, bool) else wait
#     return not cv2.waitKey(wait_ms) in [27, ord('q')] # 'ESCAPE' or 'q'

def gaussian_kernel(size, sigma):
    kernel_1d = scipy.signal.gaussian(size, sigma)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


class Model:
    """
    Convolutional neural network model.
    Class defines computation graph and provides model_fn for tf.estimator.Estimator.
    """
    def __init__(self):
        self.logger = log.getLogger('model')
        self.model_dir = 'models/cnn_v9_94x94'
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
        self.intermediate_outputs = []

    def create_layers(self):
        """This is the definition of model layers.
        It is a function as each layer can be called only once (when building model)."""
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

    def build_model(self, input, is_training=False):
        """Creates model output (logits) for given input based on model layers."""
        # batch of 1-channel images, float32, 0-255
        assert input.shape[1:3] == data.Database.IMAGE_SIZE, \
            'Something is not yes! Wrong input.shape = %s' % input.shape
        self.logger.info('Building model...')
        self.logger.info('   %s' % input.shape)
        self.intermediate_outputs = []
        self.create_layers()
        output = input
        for layer in self.layers:
            # for dropout we have to specify if it is training mode
            if isinstance(layer, tf.layers.Dropout):
                output = layer(output, training=is_training)
            else:
                output = layer(output)
            # save outputs of each layer
            self.intermediate_outputs.append(output)
            self.logger.info('   %s' % output.shape)
        return output

    def init_from_checkpoint(self):
        """Loads weights for each layer from last checkpoint.
        Needed only when not using tf.estimator.Estimator."""
        assignment_map = {}
        for layer in self.layers:
            for var in layer.variables:
                var_scope = var.name.split('/')[0]
                assignment_map['%s/' % var_scope] = '%s/' % var_scope
        # initialize all layers with weights from last checkpoint
        tf.train.init_from_checkpoint(self.model_dir, assignment_map)

    def add_histogram_summaries(self):
        """Adds histograms for all layers (weights, biases, outputs/activations)."""
        assert len(self.intermediate_outputs) == len(self.layers), 'Model not built (run build_model() first)'
        def clean(name, layer_n): # add "_0" if layer is not numbered (to have right ordering)
            if not re.match(r'^.+_\d+$', name.split('/')[0]):
                name_split = name.split('/')
                name_split[0] += '_0'
                name = '/'.join(name_split)
            return ('l%d_' % layer_n) + name.replace(':', '_')
        histograms = []
        for i, (layer, output) in enumerate(zip(self.layers, self.intermediate_outputs)):
            for weight in layer.weights: # biases and weights
                histograms.append((clean(weight.name, i), weight))
            histograms.append((clean(output.name, i), output))  # post-activation output
            try:  # try adding the pre-activation outputs (if exist)
                pre_activation_op_name = '/'.join(output.name.split('/')[:-1]) + '/BiasAdd'
                pre_activation, = output.graph.get_operation_by_name(pre_activation_op_name).outputs
                histograms.append((clean(pre_activation.name, i), pre_activation))
            except KeyError:
                pass
        for name, values in histograms:
            tf.summary.histogram(name, values)
            self.logger.debug('Adding histogram summary: %s' % name)

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
        """Model function for tf.estimator.Estimator"""
        # assemble model output from all layers
        images = features   # batch of 1-channel images, float32, 0-255
        logits = self.build_model(images, is_training=mode == ModeKeys.TRAIN)

        # outputs (loss is computed if not in predict mode)
        probabilities = tf.nn.softmax(logits)
        loss_fn = lambda : tf.losses.sparse_softmax_cross_entropy(labels, logits)

        # create summaries
        if params.get('summary_histograms', True):
            self.add_histogram_summaries()

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
        # as for now only for 1 image
        params = {'store_intermediate': True, 'store_images': True}
        predictions = self.get_estimator(params=params).predict(predict_input_fn)

        # CNN-like layers
        for prediction in predictions:
            self.logger.info('Showing input image')
            cv2_show.show_image(prediction['images'])
            for i, layer in enumerate(self.layers):
                if isinstance(layer, tf.layers.Conv2D):
                    name = layer.__class__.__name__
                    filters = prediction[i]
                    n_filters = filters.shape[-1]
                    self.logger.info('Showing layer %s nr %d - %d filters' % (name, i+1, n_filters))
                    images = np.rollaxis(filters, 2)  # move the last axis to the first one
                    to_continue = cv2_show.show_images_grid(images, normalize=True)

    def create_filter_visualizations(self, initial_image=None):
        if initial_image is not None:
            initial_images = initial_image.reshape([-1, *data.Database.IMAGE_SIZE, 1])
        else:
            initial_images = np.random.rand(1, *data.Database.IMAGE_SIZE, 1) * 255
        # create variable for the images to be optimized
        images = tf.Variable(initial_value=initial_images, name='optimized_image',
            dtype=tf.float32, constraint=lambda img: tf.clip_by_value(img, 0, 255))

        logits = self.build_model(images)
        layer_filters = self.intermediate_outputs[0]
        n_filters = layer_filters.shape[-1]
        opt_images = np.zeros([n_filters, *data.Database.IMAGE_SIZE, 1])


        for i in range(n_filters):
            loss_fn = lambda : -1 * tf.reduce_sum(layer_filters[:, :, :, i])
            optimization_op = self._optimize_image(loss_fn, images)

            self.init_from_checkpoint()
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                sess.run(optimization_op)

                opt_image = sess.run(images)[0]
                opt_images[i, :, :, :] = opt_image

        for i in range(n_filters):
            cv2_show.show_image(opt_images[i, :, :, :])

    def _optimize_image(self, loss_fn, images, blur=3, blur_sigma=3, show_img=True, show_rate=20):
        """Optimizes the image (random if not specified) to minimize given loss function"""
        # define the optimizer
        learning_rate = 1
        # as blurring is done through apply_gradients, it has to be simple GradientDescent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # gaussian kernel for blurring
        if blur is not None:
            kernel = gaussian_kernel(size=blur, sigma=blur_sigma)
            kernel = tf.constant(kernel.reshape([*kernel.shape, 1, 1]), dtype=tf.float32)
            blurred = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], 'SAME')
            blur_op = tf.assign(images, blurred)

        # loop conditions
        num_steps = 5000
        blur_each = num_steps // 10

        # image display
        def show_image_wrapper(images):
            cv2_show.show_image(images[0], wait=False)
            return np.empty(0, dtype=np.float32)  # return anything

        # optimization loop
        optimize_cond = lambda i, last_time: i < num_steps
        def optimize_body(i, last_time):
            loss = loss_fn()
            (gradient, variable), = optimizer.compute_gradients(loss, var_list=[images])
            blur_diff = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], 'SAME') - images
            gradient = tf.cond(tf.equal(i % blur_each, 0),
                lambda : gradient - blur_diff / learning_rate, # gradient is subtracted so blur_diff must be negative
                lambda : gradient)
            optimization = optimizer.apply_gradients([(gradient, variable)],
                global_step=tf.train.get_or_create_global_step())
            # image display time computations
            time_delta = tf.timestamp() - last_time
            was_long_enough = tf.greater(time_delta, 1/show_rate)
            show_image_op = tf.cond(was_long_enough,
                lambda : tf.py_func(show_image_wrapper, [images], tf.float32),
                lambda : tf.constant(0, dtype=tf.float32))
            with tf.control_dependencies([optimization]):
                if show_img:
                    with tf.control_dependencies([show_image_op]):
                        new_time = tf.cond(was_long_enough,
                            lambda : tf.timestamp(),  # if showed image, then update
                            lambda : last_time)       # else leave last_time as was
                        return i + 1, new_time
                else:
                    return i + 1, last_time
        optimize_op = tf.while_loop(optimize_cond, optimize_body, loop_vars=[tf.constant(0), tf.timestamp()])

        return optimize_op
