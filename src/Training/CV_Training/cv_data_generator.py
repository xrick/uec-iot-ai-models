#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:29:23 2024

@author: rick
"""
import os;
import sys;
import numpy as np;
import random;

sys.path.append("../");
sys.path.append("../../");

import common.utils as U

class TH_DataGenerator(object):
    def __init__(self, samples, labels, options):
        random.seed(42);
        # Initialization
        self.data = [(samples[i], labels[i]) for i in range(0, len(samples))];
        self.opt = options;
        self.batch_size = options.batchSize;
        self.preprocess_funcs = self.preprocess_setup();

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size));

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1);
        batchX = np.expand_dims(batchX, axis=3);
        return batchX, batchY

    def generate_batch(self, batchIndex):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.data[random.randint(0, len(self.data) - 1)]
                sound2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

            #For stronger augmentation
            sound = U.random_gain(6)(sound).astype(np.float32)

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);

        return sounds, labels;








