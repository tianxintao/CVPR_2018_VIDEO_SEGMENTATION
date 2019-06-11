from PIL import Image
from skimage.io import imread,imshow,show
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.transform import resize
import scipy
import torch 


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
        indexMap[index] = key
        index = index+1
        
    numOfClasses = str(len(labelmap))
#    print("The output has " + str(len(labelmap)) + " channels")
    return indexMap

# The output would be a (35,2710,3384) vector
def ConvertLabelImage(imgArr,indexMap):
    numOfClasses = len(indexMap)
    imgArray = np.array(imgArr/1000).astype(int)
#    output = np.zeros((numOfClasses,imgArray.shape[0],imgArray.shape[1]))
#    for i in range(numOfClasses):
#        output[i][imgArray - indexMap.get(i)[0] == 0] = 1
    imgArray = [list(indexMap.keys())[list(indexMap.values()).index(element)] for element in np.nditer(imgArray)]
    return imgArray


def ResizeImage(imgArr, height = 1024, width = 1024):
    originalHeight = imgArr.shape[0]
    originalWidth = imgArr.shape[1]
    ratio = width/max(originalWidth,originalHeight)
    imgArr = resize(imgArr,(round(ratio*originalHeight),round(ratio*originalWidth)),preserve_range = True)
    h,w = imgArr.shape[:2]
    max_dim = max(height,width)
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    imgArr = np.pad(imgArr, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return imgArr,ratio,window,padding

def ResizeMaskPiece(mask, scale, padding):
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)
    mask = np.pad(mask, padding[0:2], mode='constant', constant_values=0)
    return mask

def Resize(imgArr,labelArr, height = 1024, width = 1024):
    indexMap = CreateIndexMap()
    resizedImg,scale,window,padding = ResizeImage(imgArr)
    labelMask = ConvertLabelImage(labelArr,indexMap)
    resizedMask = ResizeMaskPiece(labelArr,scale,padding)
    
#    for i in range(len(indexMap)):
#        resizedMask[i] = ResizeMaskPiece(labelMask[i],scale,padding)
#        
    return resizedImg,resizedMask.astype(int)



# Convert the (35,`1024,1024) output to a (1024,1024) mask
def ConvertOutputToMask(numOfClasses,output):
    indexMap = CreateIndexMap()
    maxIndicies = torch.argmax(output,dim = 0)
    print(maxIndicies)
    numOfRows,numOfColumns = maxIndicies.shape
    for row in range(numOfRows):
        for col in range(numOfColumns):    
            maxIndicies[row][col] = indexMap[maxIndicies[row][col].item()][0]*1000
#            maxIndicies[row][col] = 0
    
    return maxIndicies
    
    
    
    
if __name__ == '__main__':
    
    sampledTrainFolderPath = '../train_color_sample/'
    sampledLabelFolderPath = '../train_label_sample/'
    
    train_images = np.array(sorted(os.listdir(sampledTrainFolderPath)))
    train_labels = np.array(sorted(os.listdir(sampledLabelFolderPath)))
    
    
    trainPic = imread(sampledTrainFolderPath+train_images[0]) 
    labelPic = imread(sampledLabelFolderPath+train_labels[0])
    imgHeight = labelPic.shape[0]
    imgWidth = labelPic.shape[1]
    
    imshow(trainPic)
    show()
    imshow(labelPic)
    show()
    
#    newPic,scale,window,padding = ResizeImage(trainPic)
#    mask = ResizeMask(ConvertLabelImage(labelPic,indexMap),scale,padding)
#    imshow(np.round(newPic))
#    imshow(mask)
    resizedImg,resizedMask = Resize(trainPic,labelPic)


