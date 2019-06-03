# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:06:41 2019

@author: douglas
"""

from PIL import Image
from skimage import io
import numpy as np
import os
from scipy.misc import imsave
from matplotlib import pyplot as plt
import shutil

def CreateIndexMap():
    # Construct a dictionary
    labelmap = {0:'others', 
                1:'rover', 
                17:'sky', 
                33:'car', 
                34:'motorbicycle', 
                35:'bicycle', 
                36:'person', 
                37:'rider', 
                38:'truck', 
                39:'bus', 
                40:'tricycle', 
                49:'road', 
                50:'siderwalk', 
                65:'traffic_cone', 
                66:'road_pile', 
                67:'fence', 
                81:'traffic_light', 
                82:'pole', 
                83:'traffic_sign', 
                84:'wall', 
                85:'dustbin', 
                86:'billboard', 
                97:'building', 
                98:'bridge', 
                99:'tunnel', 
                100:'overpass', 
                113:'vegatation', 
                161:'car_groups', 
                162:'motorbicycle_group', 
                163:'bicycle_group', 
                164:'person_group', 
                165:'rider_group', 
                166:'truck_group', 
                167:'bus_group', 
                168:'tricycle_group'}

    indexMap = dict()
    index = 0
    for key,value in labelmap.items():
        indexMap[index] = (key,value)
        index = index+1
        
    print("The output has " + str(len(labelmap)) + " channels")
    return indexMap

def GenerateMasks(image_id,input_dir): 
    image_path = os.path.join(input_dir,image_id+'.png')
    image_arr_raw = io.imread(image_path)
    image_arr_class = np.array(image_arr_raw / 1000).astype(int)
    image_arr_instances = np.array(image_arr_raw % 1000).astype(int)
    classes = np.unique(image_arr_class)
    instance_ids = np.unique(image_arr_instances)
    dir_name = os.path.join(input_dir,'masks')
    dir_name = os.path.join(dir_name,image_id + '_masks')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for c in classes:
        if c != 0:
            in_class = image_arr_class == c
            for i in instance_ids:
                if i !=255:
                    has_id = image_arr_instances == i
                    index = in_class & has_id
                    mask = np.zeros(image_arr_raw.shape,dtype = int)
                    mask[index] = 1
                    if mask.any():
                        filename = os.path.join(dir_name, str(c) + '_' + str(i)+ '.png')
                        #np.save(filename,mask)
                        imsave(filename,mask)
                        
    return dir_name

def GenerateAllMasks(input_dir):
    images = os.listdir(input_dir)
    for i in images:
        if i.endswith(".png"):
            image_id = os.path.splitext(i)[0]
            GenerateMasks(image_id,input_dir)
        
    return 
                        
                
def addToDataSet(input_dir,subset,low,high):
    image_names = os.listdir(input_dir)
    output_dir = os.path.abspath(subset)
    for i in range(low,high):
        image = os.path.join(input_dir,image_names[i])
        shutil.copy(image,output_dir)                
    
    return 
            
            
    