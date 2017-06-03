# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:51:37 2017

@author: agoswami
"""

import os
from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import isfile, join
import random
import shutil
import numpy as np

images_cooked_train_metadata_filename   = "images_cooked_train_metadata.txt" 
images_cooked_val_metadata_filename     = "images_cooked_val_metadata.txt" 
images_cooked_test_metadata_filename    = "images_cooked_test_metadata.txt" 

X_list = []
y_list = []
with open(images_cooked_train_metadata_filename, "r") as f:
    line = f.readline().strip()
    
    while(line):
        label, path = line.split(',')
#            print("{0}:{1}".format(label, path))
        
        X_list.append(imread(path))
        y_list.append(label)
        line = f.readline().strip()
    
X = np.array(X_list)
y = np.array(y_list)

#
#def loadData():
#    X_train, y_train    = loadDataFromMetadata(images_cooked_train_metadata_filename)
#    X_val, y_val        = loadDataFromMetadata(images_cooked_val_metadata_filename)
#    X_test, y_test      = loadDataFromMetadata(images_cooked_test_metadata_filename)
#    
#    return X_train, y_train, X_val, y_val, X_test, y_test
#    
#def loadDataFromMetadata(metadata_filename):
#    X_list = []
#    y_list = []
#    
#    with open(metadata_filename, "r") as f:
#        line = f.readline().strip()
#        
#        while(line):
#            label, path = line.split(',')
##            print("{0}:{1}".format(label, path))
#            
#            X_list.append(imread(path))
#            y_list.append(label)
#            line = f.readline().strip()
#        
#    X = np.array(X_list, dtype=float)
#    y = np.array(y_list, dtype=float)
#    return X, y
#
#if __name__ == "__main__":
#
#    X_train, y_train, X_val, y_val, X_test, y_test = loadData()
#    print('Train data shape: ', X_train.shape)
#    print('Train labels shape: ', y_train.shape)
#    print('Validation data shape: ', X_val.shape)
#    print('Validation labels shape: ', y_val.shape)
#    print('Test data shape: ', X_test.shape)
#    print('Test labels shape: ', y_test.shape)