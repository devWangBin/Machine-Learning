# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:09:42 2018

@author: 93568
"""
import os
from skimage import io
import numpy as np
import scipy.io as sio
import random

#数据集文件夹路径
Newspath=(r"C:\Users\93568\Documents\GitHub\Machine Learning\Gesture Recognition\...")

folders=[f for f in os.listdir(Newspath)]
print(folders)

files=[]
for folderName in  folders:
    folderPath=os.path.join(Newspath, folderName)
    files.append([f for f in os.listdir(folderPath)])

document_filenames={}
i=0

for fo in range(len(folders)):
    for fi in files[fo]:       
        document_filenames.update({i:os.path.join(Newspath,os.path.join(folders[fo],fi))})
        i+=1

#分别获取图片和标签数据
img=[]
lab=[]
for id in document_filenames:
    image = io.imread(document_filenames[id])
    label=np.zeros((6))
    
    tt = (int)(id/200)
    label[tt] = 1
    img.append(image)
    lab.append(label)

#将图片和标签数据进行打乱
c = list(zip(img, lab))
random.shuffle(c)
img[:], lab[:] = zip(*c)

print("start to make mat file!")
sio.savemat('./Hand_Gesture_image.mat', {'Hand_Gesture_Image': img})
print("image end")
sio.savemat('./Hand_Gesture_label.mat', {'Hand_Gesture_label': lab})
print("label end")