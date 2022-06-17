# -*- coding: utf-8 -*-
""" Different type of visualisation are shown in this file """
#lib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cv2

# segments following the bones networks

segments=np.array([[3, 5], [4, 6], [5, 7], [6, 8], [7,9], [8, 10], [5, 6], [5, 11], [6, 12],\
                   [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]])

# param for plotting     
font = {
'weight' : 'bold',
'size' : 14}
plt.rc('font', **font)

# plot features on a image
def plot_keypoint(self,activity,frame_number=None,FILE=None,bbox=None,with_im=False):
    
    '''
    " This function is not used in the other files but I decided to keep it as it was the function 
    used to create the visualisation shown into the report ( point/feature/segment of deadlifting)
    Input: The datapoints,bbox
    Output: the points,angles and segments on the image.

    '''
    if with_im==True:
        FILE_IMAGE_T=FILE.replace(".json", ".jpg")
        FILE_IMAGE_T=FILE_IMAGE_T.replace("Json", "Image")
        _ , _ , w, h=bbox
        ratio_hw=h/w
        bbox=bbox.astype(np.int64)
        print(FILE_IMAGE_T)
        image_frame = cv2.imread(FILE_IMAGE_T,cv2.IMREAD_COLOR)
        im=image_frame[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[0]+bbox[2],:]
        success=True
        self.x=self.x*im.shape[1]
        self.y=self.y*im.shape[0]/ratio_hw
    
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    # keypoints
    plt.subplot(1,3,1)
    plt.scatter(self.x,self.y,s=72)
    for j in range(0,self.shape[0]):
        if (pd.isna(self.x[j]) and pd.isna(self.y[j])) is False:
            plt.text(self.x[j],self.y[j],str(j),color="red")
    for i,[p1,p2] in enumerate(segments):
        if (pd.isna(self.x[p1]) and pd.isna(self.y[p1]) and pd.isna(self.x[p2]) and pd.isna(self.y[p2])) is False:
            plt.plot([self.x[p1], self.x[p2]], [self.y[p1], self.y[p2]], linewidth=4)
            plt.title("Point",font)
            if with_im==True and success==True:
                plt.imshow(im)
            
    #segment
    plt.subplot(1,3,2)
    plt.scatter(self.x,self.y,s=72)
    for i,[p1,p2] in enumerate(segments):
        if (pd.isna(self.x[p1]) and pd.isna(self.y[p1]) and pd.isna(self.x[p2]) and pd.isna(self.y[p2])) is False:
            plt.plot([self.x[p1], self.x[p2]], [self.y[p1], self.y[p2]], linewidth=4)
            plt.text(0.5*(self.x[p1]+self.x[p2]), 0.5*(self.y[p1]+self.y[p2]), "J"+str(i+1),color="red")
            plt.title("Segment",font)
            if (with_im==True) and (success==True):
                plt.imshow(im)
            
    #angle
    plt.subplot(1,3,3)
    plt.scatter(self.x,self.y,s=72)
    angles_human=np.array([[3, 5,6], [4, 6,5], [6, 5,11], [5, 6,12], [7,5,11], [8,6,12], [5, 7,9], [6,8,10], [5,11,12],\
                   [6,12,11], [13,11,12], [14,12,11], [15,13,11], [16,14,12]])
    for i,[p1,p2] in enumerate(segments):
        if (pd.isna(self.x[p1]) and pd.isna(self.y[p1]) and pd.isna(self.x[p2]) and pd.isna(self.y[p2])) is False:
            plt.plot([self.x[p1], self.x[p2]], [self.y[p1], self.y[p2]], linewidth=4)
    for i,[p1,p2,p3] in enumerate(angles_human):
        if (pd.isna(self.x[p1]) and pd.isna(self.y[p1]) and pd.isna(self.x[p2]) and pd.isna(self.y[p2])and pd.isna(self.x[p3]) and pd.isna(self.y[p3])) is False:
            m1=0.5*(self.x[p1]+self.x[p2]), 0.5*(self.y[p1]+self.y[p2])
            m2=0.5*(self.x[p2]+self.x[p3]), 0.5*(self.y[p2]+self.y[p3])
            plt.text(0.5*(m1[0]+m2[0]), 0.5*(m1[1]+m2[1]), "a"+str(i+1),horizontalalignment='center',verticalalignment='center',color="red")
            plt.title("Angle",font)
            if (with_im==True) and success==True:
                plt.imshow(im)
    plt.suptitle(activity,font=font)
    plt.show()

#------------------------------------------------------------------------------------

# visu for videos 

def print_frame(FILE,frame,sp,labels_size):

    '''
    Input: the files,the frames and the lable size ( depends on how much image we want to output)
    Output: plot the different chosen image from a video ( used for the video)
    
    '''
    FILE=FILE.replace("_openpifpaf.json", ".mp4")
    vidcap = cv2.VideoCapture(FILE)
    success,image = vidcap.read()
    count = 0
    while success:
        success,image = vidcap.read()
        if count==frame:
            image_frame=image  
            im=cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            plt.subplot(1,labels_size,sp)
            plt.imshow(im,cmap="gray")
            plt.title("Frame "+str(frame))
            break
        count += 1


#------------------------------------------------------------------------------------