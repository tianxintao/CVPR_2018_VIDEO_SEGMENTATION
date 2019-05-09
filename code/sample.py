import os
import numpy as np
from PIL import Image
#print(os.listdir("../"))


train_color_dir = '../train_color'
train_label_dir = '../train_label'
test_dir = '../test'

train_images = np.array(sorted(os.listdir(train_color_dir)))
train_labels = np.array(sorted(os.listdir(train_label_dir)))
test_images = np.array(sorted(os.listdir(test_dir)))




def SampleImagesAndLabels(size,save = False, showTrain = False, showLabel = False):
    indexArray = np.random.randint(0,len(train_images)-1,size)
    selectedImages = train_images[indexArray]
    selectedLables = [imgName.replace('.jpg','_instanceIds.png',1) for imgName in selectedImages]
    
#    selectedImagesPath = [train_color_dir+"/"+imgName for imgName in selectedImages]
#    selectedLabelPath = [train_label_dir+"/"+labelName for labelName in selectedLables]
    
    for img in selectedImages:
        selectedImagePath = train_color_dir+"/"+img
        labelImageName = img.replace('.jpg','_instanceIds.png',1)
        im = Image.open(selectedImagePath)
        imLabel = Image.open(train_label_dir+"/"+labelImageName)
        
        if(showTrain):
            im.show()
        if(showLabel):
            imLabel.show()
        
        newTrainFolderPath = '../train_color_sample/'
        newLabelFolderPath = '../train_label_sample/'
        if not os.path.exists(newTrainFolderPath):
            os.mkdir(newTrainFolderPath)
        if not os.path.exists(newLabelFolderPath):
            os.mkdir(newLabelFolderPath)
        im.save(newTrainFolderPath+img)
        imLabel.save(newLabelFolderPath+labelImageName,"PNG")
        
        
#        print("Done")
#        im.save(img.replace('../train_color','../train_color_sample')