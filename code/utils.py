from PIL import Image
from skimage.io import imread,imshow,show
import numpy as np
import os
from matplotlib import pyplot as plt

sampledTrainFolderPath = '../train_color_sample/'
sampledLabelFolderPath = '../train_label_sample/'

train_images = np.array(sorted(os.listdir(sampledTrainFolderPath)))
train_labels = np.array(sorted(os.listdir(sampledLabelFolderPath)))


trainPic = imread(sampledTrainFolderPath+train_images[2]) 
labelPic = imread(sampledLabelFolderPath+train_labels[2])
imgHeight = labelPic.shape[0]
imgWidth = labelPic.shape[1]

imshow(trainPic)
show()
imshow(labelPic)
show()

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

indexMap = labelmap.copy()
index = 0
for key,value in indexMap.items():
    indexMap[key] = index
    index = index+1
    
numOfClasses = str(len(labelmap))
print("The output has " + str(len(labelmap)) + " channels")

# The output would be a (35,2710,3384) vector
def ConvertLabelImage(imgArr,numOfClasses):
    imgArray = np.array(imgArr/1000).astype(int)
    output = np.zeros((numOfClasses,imgArr.shape[0],imgArr.shape[1]))
    return imgArray
    



