# -*- coding: utf-8 -*-
"""
Created on Tue May 16 05:58:18 2017

@author: agoswami
"""
import os
from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import isfile, join
import random
import shutil

images_raw_path = "images_raw/"
images_cooked_train_path    = "images_cooked_train/"
images_cooked_val_path      = "images_cooked_val/"
images_cooked_test_path     = "images_cooked_test/"

#delete cooked directories if they exist
if os.path.exists(images_cooked_train_path):
    shutil.rmtree(images_cooked_train_path)

if os.path.exists(images_cooked_val_path):
    shutil.rmtree(images_cooked_val_path)

if os.path.exists(images_cooked_test_path):
    shutil.rmtree(images_cooked_test_path)

images_cooked_train_metadata_filename   = "images_cooked_train_metadata.txt" 
images_cooked_val_metadata_filename     = "images_cooked_val_metadata.txt" 
images_cooked_test_metadata_filename    = "images_cooked_test_metadata.txt" 

labelsDict = {
        'cats':0,
        'dogs':1,
        'giraffe':2}

with open(images_cooked_train_metadata_filename, "w") as f_train, \
   open(images_cooked_val_metadata_filename, "w") as f_val, \
   open(images_cooked_test_metadata_filename, "w") as f_test:
    
    for label in labelsDict:
        images_raw_path_withlabel = images_raw_path + label
        
        filenames = [f for f in listdir(images_raw_path_withlabel) if isfile(join(images_raw_path_withlabel, f))]
        print(filenames)
        
        for file in filenames:
            file = images_raw_path_withlabel + "/" + file
            
            img = imread(file)
            imgresize = imresize(img, (32, 32))
    
            _r = random.random()
            print(_r)
            if (_r < 0.8):
        #        save resized image in 'images_cooked_train' folder, and make entry in metadata file
                file_new = file.replace(images_raw_path, images_cooked_train_path)
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
                    
                imsave(file_new, imgresize)
                f_train.write(str(labelsDict[label]) + "," + file_new + "\n")
            elif (_r > 0.8 and _r < 0.9):    
        #        save resized image in 'images_cooked_val' folder, and make entry in metadata file
                file_new = file.replace(images_raw_path, images_cooked_val_path)
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
                    
                imsave(file_new, imgresize)
                f_val.write(str(labelsDict[label]) + "," + file_new + "\n")
            else:
        #        save resized image in 'images_cooked_test' folder, and make entry in metadata file
                file_new = file.replace(images_raw_path, images_cooked_test_path)
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
        
                imsave(file_new, imgresize)
                f_test.write(str(labelsDict[label]) + "," + file_new + "\n")


#with open("images_raw_metadata.txt", "r") as f_in, \
#    open("images_cooked_metadata.txt", "w") as f_out:
#    
#    header = f_in.readline().strip()
#    f_out.write(header + "\n")
##    print(header)
#    
#    line = f_in.readline().strip()
#    while(line):
#        c, path_raw = line.split(',')
#        print("{0}:{1}".format(c, path_raw))
#        
#        img = imread(path_raw)
##        print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"
#        
#        path_cooked = path_raw.replace('images_raw', 'images_cooked')
#        directory = os.path.dirname(path_cooked)
#        if not os.path.exists(directory):
#            os.makedirs(directory)
#            
#        imsave(path_cooked, imresize(img, (32, 32)))
#        f_out.write(c + "," + path_cooked + "\n")
        
#        line = f_in.readline().strip()
    