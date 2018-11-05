#!/usr/bin/env python3

#-----------------------------------------------------------------------------------------------
# This file is a part of the final project for CAP5768
#
# Functions to make operations on images
#
#   Functions list:
#       * read_image(files, desired_a)
#           Input:
#               list of files and desired width of resized image;
#           Objective:
#               1) Read an image as an numpy array
#               2) Resize an image to square with length of desired_a
#
#       * get_tensor(path, train_size, test_size, batch_size, desired_shape=300)
#           Input:
#               path - path with training image folder
#               train_size - number of images to be in training set
#               test_size - number of images to be in test set
#               batch_size - size of batch for tensorflow training
#               desired_shape - desired width of resized image
#           Output:
#               batch train and test data in tf.data format
#
# DEPENDENCIES:
#   * numpy
#   * tensorflow
#   * OpenCV
#-----------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import os
import cv2
from random import shuffle

def read_image(path, files, desired_a):
    # Cat vs. Dog dataset https://www.kaggle.com/c/dogs-vs-cats
    # fesired_a - desired length of a side of square
    pixels = []
    labels = []
    for img_name in files:
        #try:
        # Read the label of image:
        if img_name != ".DS_Store":
            try:
                lbl = img_name.split('.')[-3].split("/")[-1]
            except IndexError:
                print("Error. Can't read label for " + img_name)
                exit()
            if lbl == 'cat':
                lbl = [1, 0]
            else:
                lbl = [0, 1]
            # Read image in greyscale:
            img_pix = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            old_size = img_pix.shape[:2]  # old size in (height, width) format
            ratio = desired_a / max(old_size)
            # Make height or width equal to desired_a
            new_size = tuple([int(x * ratio) for x in old_size])
            img_pix = cv2.resize(img_pix, (new_size[1], new_size[0]))
            delta_w = desired_a - new_size[1]
            delta_h = desired_a - new_size[0]
            # Create padding to make the other side equal to a_desired:
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            new_im = cv2.copyMakeBorder(img_pix, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=[0])
            new_im = cv2.resize(new_im, (desired_a, desired_a))
            # Append resized image and label to the list
            pixels.append(new_im)
            labels.append(lbl)
       # except AttributeError:#if not a catdog file in folder
          #  pass
    return np.array(pixels), np.array(labels)

def get_tensor(path, train_size, test_size, batch_size, desired_shape=300):
    # get data tensor form cat/dog dataset
    # 0 - dog; 1 - cat
    # resize all images to square 300x300 with zero padding to save original ratio
    files = os.listdir(path)
    shuffle(files)
    train_files = files[0:train_size]
    test_size = files[train_size:train_size+test_size]
    pixels_train, labels_train = read_image(path, train_files, desired_shape)
    pixels_test, labels_test = read_image(path, test_size, desired_shape)
    train_data = tf.data.Dataset.from_tensor_slices((pixels_train, labels_train))
    test_data = tf.data.Dataset.from_tensor_slices((pixels_test, labels_test))
    train_data = train_data.batch(batch_size)
    test_data = test_data.batch(batch_size)
    return train_data, test_data

#read_catdog(["/Users/stebliankin/Desktop/Data Science-CAP5768/project/all/train/cat.0.jpg"], 300)
#get_catdog_tensor("/Users/stebliankin/Desktop/Data Science-CAP5768/project/all/train/",100,10,5)