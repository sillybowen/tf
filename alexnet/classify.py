#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import json
import logging
from datetime import datetime
import time

import StringIO
import cStringIO
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import alexnet_v1
slim = tf.contrib.slim

class AlexNet(object):

    def __init__(self, ckpt_path, num_classes=1000, label_map_file='label.map', gpu_id=0):
        self.logger = logging.getLogger(__name__)
        self.pred_op = None
        self.sess = None
        self.net_data=None
        self.checkpoint_path = ckpt_path
        self.gpu_id = gpu_id
        if label_map_file:
            self.human_labels = self.read_human_readable_label(label_map_file)

        self._build_graph(self.gpu_id)

    def read_human_readable_label(self, label_map_file):
        human_labels = map(lambda line: line.strip(), open(label_map_file))
        human_labels = dict(map(lambda i: (i, human_labels[i]), xrange(len(human_labels))))
        return human_labels

    def my_softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def _build_graph(self, gpu_id):
        """Build the inception graph
        Args:
            gpu_id: indicate which gpu is to use
        """
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                inputs = tf.placeholder(tf.float32,
                                        [None, 224, 224, 3],
                                        name='input')
#                self.pred_op, _ = alexnet_v1.alexnet_v2(inputs, num_classes=1000, is_training=False)
                self.pred_op_numpy, _ = alexnet_v1.alexnet_v1(inputs, num_classes=1000, is_training=False)
#                self.w_op = alexnet_v1.get6W()
            # Create a session and restore weights
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True,
                device_count={"CPU": 4},
#                gpu_options=gpu_options,
                intra_op_parallelism_threads=4,
                inter_op_parallelism_threads=4               
            )
            self.sess = tf.Session(config=sess_config)

            self.logger.info("restore checkpoint file")
            restore_vars = tf.trainable_variables()
            self.restore_alexnet_from_npy(self.checkpoint_path, restore_vars)
            self.logger.info('{}: checkpoint file restored from {}'.format(datetime.now(),
                                                                              self.checkpoint_path))

    def restore_alexnet_from_npy(self, file_name, variables):
        alexnet_map = {
            "alexnet_v1/conv1_w:0": ["conv1", 0],
            "alexnet_v1/conv1_b:0": ["conv1", 1],
            "alexnet_v1/conv2_w:0": ["conv2", 0],
            "alexnet_v1/conv2_b:0": ["conv2", 1],
            "alexnet_v1/conv3_w:0": ["conv3", 0],
            "alexnet_v1/conv3_b:0": ["conv3", 1],
            "alexnet_v1/conv4_w:0": ["conv4", 0],
            "alexnet_v1/conv4_b:0": ["conv4", 1],
            "alexnet_v1/conv5_w:0": ["conv5", 0],
            "alexnet_v1/conv5_b:0": ["conv5", 1],
            "alexnet_v1/fc6_w:0": ["fc6", 0],
            "alexnet_v1/fc6_b:0": ["fc6", 1],
            "alexnet_v1/fc7_w:0": ["fc7", 0],
            "alexnet_v1/fc7_b:0": ["fc7", 1],
            "alexnet_v1/fc8_w:0": ["fc8", 0],
            "alexnet_v1/fc8_b:0": ["fc8", 1]
        }
        self.net_data = np.load(file_name).item()
        print (type(np.load(file_name)))

        for variable in variables:
            if variable.name in alexnet_map.keys():
                key, index = alexnet_map[variable.name][0], alexnet_map[variable.name][1]
                self.sess.run(variable.assign(self.net_data[key][index]))

    def preprocess(self, image_buffer):
        im = Image.open(cStringIO.StringIO(image_buffer)).convert('RGB').resize((224, 224))
        im = np.array(im).astype(np.float32)
        im = im - np.mean(im)
        im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
        return im

    def classify(self, image_data, top_k=1):
        """Run the graph for classification,
        Args:
            1. image_data: one raw image data or batch of preprocessed image data
            3. top_k: return top k classifications
        Returns:
            a top k or a list of top k classifications of format [{'name': cat, 'weight': 0.2}, {'name': 'animal', 'weight': xx}]
        """
        if not isinstance(image_data, list):
            image_data = [image_data]
        images = []
        for image in image_data:
            images.append(self.preprocess(image))
        start_time = time.time()
 #       run_metadata = tf.RunMetadata()
        fc6x = self.sess.run(self.pred_op_numpy, {'input:0': images},
 #                                   options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
 #                                   run_metadata=run_metadata
        )
        print('cov time: %s' % (time.time() - start_time))
        fc6 = np.dot(fc6x, self.net_data['fc6'][0])+ self.net_data['fc6'][1]
        print('matmul6 time: %s' % (time.time() - start_time))
        fc7x = np.maximum(fc6,0)
        print('relu6 time: %s' % (time.time() - start_time))
        fc7 = np.dot(fc7x, self.net_data['fc7'][0])+ self.net_data['fc7'][1]
        print('matmul7 time: %s' % (time.time() - start_time))
        fc8x = np.maximum(fc7,0)
        print('relu7 time: %s' % (time.time() - start_time))
        fc8 = np.dot(fc8x, self.net_data['fc8'][0])+ self.net_data['fc8'][1]
        print('matmul8 time: %s' % (time.time() - start_time))
  #      softmaxop = alexnet_v1.softmax(fc8)
 #       newsession = tf.Session()
#        predictions =newsession.run(softmaxop)
        predictions = self.my_softmax(fc8)
        print('softmax time: %s' % (time.time() - start_time))
#        print(predictions)
        
#        fetched_timeline=timeline.Timeline(run_metadata.step_stats)
#        chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
#        with open('timeline_01.json', 'w') as f:
#            f.write(chrome_trace)

#        return 1
        return map(lambda pred: self.get_top_k_classes(pred, top_k), predictions)

    def get_top_k_classes(self, prediction, k):
        """given the predictions returned by classify, you can get the top k classes
        that this image may belong to """
        index = prediction.argsort()[-k:][::-1]
        classes = [{"name": self.human_labels.get(i, "unknown"), "weight": float(prediction[i])} for i in index if prediction[i] > 0.0]
        return classes

    #    def classify2(self, image_data, top_k=1):
        """Run the graph for classification,
        Args:
            1. image_data: one raw image data or batch of preprocessed image data
            3. top_k: return top k classifications
        Returns:
            a top k or a list of top k classifications of format [{'name': cat, 'weight': 0.2}, {'name': 'animal', 'weight': xx}]
        """
"""        if not isinstance(image_data, list):
            image_data = [image_data]
        images = []
        for image in image_data:
            images.append(self.preprocess(image))
        start_time = time.time()
 #       run_metadata = tf.RunMetadata()
        predictions = self.sess.run(self.pred_op, {'input:0': images},
 #                                   options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
 #                                   run_metadata=run_metadata
        )
#        predictions = self.my_softmax(fc8)
#        print(predictions)
        
#        fetched_timeline=timeline.Timeline(run_metadata.step_stats)
#        chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
#        with open('timeline_01.json', 'w') as f:
#            f.write(chrome_trace)

#        return 1
        return map(lambda pred: self.get_top_k_classes(pred, top_k), predictions)    """


if __name__ == '__main__':
    prepare_time = time.time()
    logging.basicConfig(level='INFO', format='%(levelname)s:%(message)s')
    model = AlexNet('./models/bvlc_alexnet.npy', label_map_file='data/label.map', gpu_id=0)
    file_name = 'data/cat.jpg'
    image_buffer = open(file_name).read()
    print('prepare time: %s' % (time.time() - prepare_time))
    start_time = time.time()
#    model.classify2(image_buffer, top_k=10)
    print('classification result: %s' % model.classify(image_buffer, top_k=10))
#    print('spend time: %s' % (time.time() - start_time))
#    sigma=0
#    for i in range(1,100):
#        start_time = time.time()
 #       model.classify(image_buffer, top_k=10)
#        print('classification result: %s' % model.classify(image_buffer, top_k=10))
#        print('spend time: %s' % (time.time() - start_time))
#        sigma = sigma + time.time() - start_time
#    sigma = sigma/100
#    print(sigma)

    #start_time = time.time()
    #print('classification result: %s' % model.classify(image_buffer, top_k=1))
    #print('spend time: %s' % (time.time() - start_time))

    #start_time = time.time()
    #print('classification result: %s' % model.classify(image_buffer, top_k=1))
    #print('spend time: %s' % (time.time() - start_time))



