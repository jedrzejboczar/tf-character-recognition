#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time

import scipy.signal
import numpy as np
import tensorflow as tf

import log
import data
import cv2_show


ModeKeys = tf.estimator.ModeKeys  # shorter

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
        self.model_dir = 'models/cnn_v13_94x94'
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.intermediate_outputs = []

    def create_layers(self):
        """This is the definition of model layers.
        It is a function as each layer can be called only once (when building model)."""
        self.layers = [
            # Convolutional part
            tf.layers.Conv2D(filters=8, kernel_size=3, activation=tf.nn.relu),   # [92, 92, 8]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [46, 46, 8]
            tf.layers.SeparableConv2D(filters=32, kernel_size=3, activation=tf.nn.relu),  # [44, 44, 8]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [22, 22, 8]
            tf.layers.SeparableConv2D(filters=72, kernel_size=3, activation=tf.nn.relu),  # [20, 20, 32]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [10, 10, 32]
            tf.layers.SeparableConv2D(filters=160, kernel_size=3, activation=tf.nn.relu), # [8, 8, 128]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [4, 4, 128]
            tf.layers.SeparableConv2D(filters=512, kernel_size=3, activation=tf.nn.relu), # [2, 2, 512]
            tf.layers.MaxPooling2D(pool_size=2, strides=2),                      # [1, 1, 512]
            # Dense part
            tf.layers.Flatten(),                                                 # [512]
            tf.layers.Dense(units=178, activation=tf.nn.relu),                  # [178]
            tf.layers.Dropout(rate=.8),
            tf.layers.Dense(units=data.Database.N_CLASSES),
        ]

    def build_model(self, input, is_training=False, build_layers=True):
        """Creates model output (logits) for given input based on model layers."""
        # batch of 1-channel images, float32, 0-255
        assert input.shape[1:3] == data.Database.IMAGE_SIZE, \
            'Something is not yes! Wrong input.shape = %s' % input.shape
        info = lambda name, shape, n_train: \
            self.logger.info('  %20s -> %18s #%d' % (name, shape, n_train))
        self.logger.info('Building model...')
        self.intermediate_outputs = []
        if build_layers:
            self.create_layers()
        info('input images', input.shape, 0)
        output = input
        for layer in self.layers:
            # for dropout we have to specify if it is training mode
            if isinstance(layer, tf.layers.Dropout):
                output = layer(output, training=is_training)
            else:
                output = layer(output)
            # save outputs of each layer
            self.intermediate_outputs.append(output)
            n_trainable = sum(w.shape.num_elements() for w in layer.weights)
            info(layer.name, output.shape, n_trainable)
        # count number of model parameters
        n = sum(w.shape.num_elements() for l in self.layers for w in l.weights)
        self.logger.info('total number of trainable parameters is #%d' % n)
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
        if params.get('summary_histograms', True) and mode == ModeKeys.TRAIN:
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

    def visualize_activations(self, predict_input_fn):
        """Show outputs of intermediate layers for data from predict_input_fn()"""
        # as for now only for 1 image
        params = {'store_intermediate': True, 'store_images': True}
        predictions = self.get_estimator(params=params).predict(predict_input_fn)
        for prediction in predictions:
            self.logger.info('Showing input image')
            cv2_show.show_image(prediction['images'], resize_to_fit=True)
            for i, layer in enumerate(self.layers):
                to_continue = True
                # CNN-like layers
                if isinstance(layer, (tf.layers.Conv2D, tf.layers.MaxPooling2D)):
                    filters = prediction[i]
                    n_filters = filters.shape[-1]
                    self.logger.info('Showing layer %s - %d filters' % (layer.name, n_filters))
                    images = np.rollaxis(filters, 2)  # move the last axis to the first one
                    to_continue = cv2_show.show_images_grid(images, normalize=True, visualize_negative=True)
                # dense layers (1D)
                elif isinstance(layer, tf.layers.Dense):
                    activations = prediction[i]
                    self.logger.info('Showing layer %s - %d values (reshaped into grid)' \
                        % (layer.name, activations.shape[0]))
                    images = activations[:, None, None]
                    # totally inefficient, but here it is really not needed
                    to_continue = cv2_show.show_images_grid(images, normalize=True, visualize_negative=True)
                if not to_continue:
                    break

    def create_filter_visualizations(self, layer_num, initial_image=None):
        self.create_layers()
        layer = self.layers[layer_num]
        assert isinstance(layer, tf.layers.Conv2D), 'Only Conv2D layers supported'

        if initial_image is not None:
            initial_images = initial_image.reshape([-1, *data.Database.IMAGE_SIZE, 1])
        else:
            initial_images = np.random.rand(1, *data.Database.IMAGE_SIZE, 1) * 255
        # create variable for the images to be optimized
        images_vars = [tf.Variable(initial_value=initial_images, name='optimized_image',
            dtype=tf.float32, constraint=lambda img: tf.clip_by_value(img, 0, 255))
            for _ in range(layer.filters)]
        optimizations = []
        losses = []

        for filter_num in range(layer.filters):
            images = images_vars[filter_num]
            logits = self.build_model(images, build_layers=False)
            filters = self.intermediate_outputs[layer_num]
            # loss_fn = lambda : -1 * tf.reduce_sum(filters[0, :, :, filter_num])
            loss_fn = lambda : -1 * tf.reduce_sum(filters[0, :, :, filter_num])
            optimization_op = self.optimize_image(loss_fn, images, num_steps=5000,
                learning_rate=1, blur=3, blur_each=500, show_img=False)
            optimizations.append(optimization_op)
            losses.append(loss_fn)

        self.init_from_checkpoint()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            self.logger.info('Optimizing input images...')
            start = time.time()
            sess.run(optimizations)
            self.logger.info('...finished in %.3f seconds' % (time.time() - start))
            losses = sess.run([loss() for loss in losses])
            self.logger.info('Losses: %s' % (losses))
            optimized_images = sess.run(images_vars)

        optimized_images = np.stack(optimized_images).squeeze(axis=1)
        cv2_show.show_images_grid(optimized_images, resize_to_fit=False)

        # for i in range(n_filters):
        #     cv2_show.show_image(opt_images[i, :, :, :])

    def optimize_image(self, loss_fn, images, num_steps, learning_rate=1, blur=None, blur_sigma=3, blur_each=None, show_img=True, show_rate=10):
        """Optimizes the image to minimize given loss function"""
        blur_each = blur_each if blur_each else num_steps // 2
        # as blurring is done through apply_gradients, it has to be simple GradientDescent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # gaussian kernel for blurring
        if blur:
            kernel = gaussian_kernel(size=blur, sigma=blur_sigma)
            kernel = tf.constant(kernel.reshape([*kernel.shape, 1, 1]), dtype=tf.float32)
            blur = lambda imgs: tf.nn.conv2d(imgs, kernel, [1, 1, 1, 1], 'SAME')
        # image display
        def show_image_wrapper(images):
            cv2_show.show_image(images[0], wait=False)
            return np.empty(0, dtype=np.float32)  # return anything
        # optimization loop
        optimize_cond = lambda it, last_time: it < num_steps
        def optimize_body(it, last_time):
            loss = loss_fn()
            (gradient, variable), = optimizer.compute_gradients(loss, var_list=[images])
            if blur:
                blur_diff = blur(images) - images
                to_blur = tf.equal(it % blur_each, 0)
                gradient = tf.cond(to_blur,
                    lambda : gradient - blur_diff / learning_rate, # gradient is subtracted so blur_diff must be negative
                    lambda : gradient)
            optimization = optimizer.apply_gradients([(gradient, variable)],
                global_step=tf.train.get_or_create_global_step())
            # image display-time computations
            time_delta = tf.timestamp() - last_time
            was_long_enough = tf.greater(time_delta, 1/show_rate)
            show_image_op = tf.cond(was_long_enough,
                lambda : tf.py_func(show_image_wrapper, [images], tf.float32),
                lambda : tf.constant(0, dtype=tf.float32))
            with tf.control_dependencies([optimization]):
                it = it + 1
                if show_img:
                    with tf.control_dependencies([show_image_op]):
                        new_time = tf.cond(was_long_enough,
                            lambda : tf.timestamp(),  # if showed image, then update
                            lambda : last_time)       # else leave last_time as was
                        return it, new_time
                else:
                    return it, last_time
        optimize_op = tf.while_loop(optimize_cond, optimize_body, loop_vars=[tf.constant(0), tf.timestamp()])
        return optimize_op

###
# enc_v1 - MSE, relu activation for last layer
# enc_v2 - MSE, sigmoid activation * 255
###
class Autoencoder:
    def __init__(self):
        self.logger = log.getLogger('autoencoder')
        self.model_dir = 'models/enc_v2'
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.encoded = None

    def get_estimator(self, **kwargs):
        """Creates an instance of tf.estimator.Estimator for the model.

        If given, passes all keyword arguments to its constructor, else
        it uses model's default values."""
        return tf.estimator.Estimator(
            self.model_fn,
            model_dir=kwargs.get('model_dir', self.model_dir),
            **kwargs,
        )

    def build_model(self, input, is_training=False):
        """Creates output for given input. Saves encoded part."""
        # batch of 1-channel images, float32, 0-255
        assert input.shape[1:3] == data.Database.IMAGE_SIZE, \
            'Something is not yes! Wrong input.shape = %s' % input.shape

        self.logger.info('Building model...')
        self.logger.info('  %18s  (input)' % (input.shape))
        encoded = self.build_encoder(input, is_training=is_training)
        decoded = self.build_decoder(encoded, is_training=is_training)
        self.logger.info('  %18s  (output)' % (input.shape))

        self.logger.info('Trainable variables:')
        n = 0
        for var in tf.get_collection('trainable_variables'):
            if var.name.startswith('global_step'):
                continue
            n += var.shape.num_elements()
            self.logger.info('  #%-8d (%s)' % (var.shape.num_elements(), var.name))
        self.logger.info('total number of parameters: %d' % n)

        return encoded, decoded

    def init_from_checkpoint(self):
        """Loads weights for each layer from last checkpoint.
        Needed only when not using tf.estimator.Estimator."""
        assignment_map = {}
        for var in tf.get_collection('trainable_variables'):
            var_scope = var.name.split('/')[0]
            assignment_map['%s/' % var_scope] = '%s/' % var_scope
        tf.train.init_from_checkpoint(self.model_dir, assignment_map)

    def model_fn(self, features, labels, mode, config=None, params={}):
        """Model function for tf.estimator.Estimator"""
        # assemble model output from all layers
        images = features   # batch of 1-channel images, float32, 0-255
        encoded, reconstructed = self.build_model(images, is_training=mode == ModeKeys.TRAIN)
        # loss is computed if not in predict mode
        loss = tf.losses.mean_squared_error(images, reconstructed)
        # create EstimatorSpecs depending on mode
        if mode == ModeKeys.PREDICT:
            predictions = {
                'reconstructed': reconstructed,
            }
            return tf.estimator.EstimatorSpec(mode, predictions)
        if mode == ModeKeys.TRAIN:
            grads_and_vars = self.optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(grads_and_vars):
                tf.summary.histogram('gradient_%s' % var.name, grad)
            optimization = self.optimizer.apply_gradients(grads_and_vars,
                global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimization)
        assert mode == ModeKeys.EVAL, 'Received unexpected mode: %s' % mode
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    def build_encoder(self, input, is_training=False):
        output = input
        output = tf.layers.conv2d(output, filters=8, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d(output, filters=8, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d(output, filters=16, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d(output, filters=16, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        return output

    def build_decoder(self, input, is_training=False):
        output = input
        output = self.upscaling2d(output, times=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d_transpose(output, filters=16, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = self.upscaling2d(output, times=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d_transpose(output, filters=16, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = self.upscaling2d(output, times=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d_transpose(output, filters=8, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = self.upscaling2d(output, times=2)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d_transpose(output, filters=8, kernel_size=5, activation=tf.nn.relu)
        self.logger.info('  %18s  (%s)' % (output.shape, output.name))
        output = tf.layers.conv2d_transpose(output, filters=1, kernel_size=3, activation=tf.nn.sigmoid)
        return output * 255

    def max_unpooling2d(self, input, indicies):
        # should use indicies from tf.nn.max_pool_with_argmax to put values in correct places
        raise NotImplementedError()

    def upscaling2d(self, input, times=2):
        batch, height, width, channels = input.shape.as_list()  # must be 4D
        new_height, new_width = times * height, times * width
        # method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        upscaled = tf.image.resize_images(input, [new_height, new_width], method)
        return upscaled

    def walk_latent_space(self, path_intermediate_images, n_per_step):
        """Generates images on the path by linear interpolations between path points."""
        # find points in the latent space
        path_intermediate_points = self.build_encoder(tf.stack(path_intermediate_images))
        # create linear interpolation of path
        path = []
        for i in range(path_intermediate_points.shape[0] - 1):
            start_point = path_intermediate_points[i]
            end_point = path_intermediate_points[i + 1]
            distance = end_point - start_point
            path.extend([start_point + j/n_per_step * distance for j in range(n_per_step + 1)])
        # put images into batches (to avoid problems with memory)
        path = tf.data.Dataset.from_tensor_slices(tf.stack(path))
        path = path.batch(n_per_step).prefetch(1)
        # create the iterator
        path_iterator = path.make_initializable_iterator()
        next_path_batch = path_iterator.get_next()
        # create image representations of points in the path
        decoded_batch = self.build_decoder(next_path_batch)
        # overwrite initialization of weights with trained ones
        self.init_from_checkpoint()
        with tf.Session() as sess:
            # initializations
            sess.run(tf.global_variables_initializer())
            sess.run(path_iterator.initializer)
            # iterate over data yielding batches
            while True:
                try:
                    images = sess.run(decoded_batch)
                    yield images
                except tf.errors.OutOfRangeError:
                    break
