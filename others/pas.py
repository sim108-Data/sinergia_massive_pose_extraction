# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import others.visualisation as kp_visu
import others.helpers as hp
import matplotlib.pyplot as plt
import cv2


def separate_body_part(FILE,type):

    '''
    Input: path of the file (FILE) [string], type of normalisation (type) [string]
    Output: data set for each 6 part of the body and also the bbox 
    
    '''

    keypoints,bbox = hp.get_pose_fromJson_JPG(FILE)
    # normalise the point on bbox
    keypointCenterBbox = hp.normalise(keypoints,bbox, type = type)
    # Séparation en 6 partie 
    pointFeaturesHead = keypointCenterBbox.loc[0:4,:]
    pointFeaturesArmLeft = keypointCenterBbox.loc[[5,7,9],:]
    pointFeaturesArmRight = keypointCenterBbox.loc[[6,8,10],:]
    pointFeaturesTorso = keypointCenterBbox.loc[[5,6,11,12],:]
    pointFeaturesLegLeft = keypointCenterBbox.loc[[11,13,15],:]
    pointFeaturesLegRight = keypointCenterBbox.loc[[12,14,16],:]
    return pointFeaturesHead,pointFeaturesArmLeft,pointFeaturesArmRight,pointFeaturesTorso,pointFeaturesLegLeft,pointFeaturesLegRight,bbox,keypointCenterBbox

#------------------------------------------------------------------------------------
# The following function have been find at this url : 
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

def get_line(p1, p2):

    '''
    Input take point as a list of coord
    Output the line between the two point
    '''
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def get_intersection(L1, L2):

    '''
    Input: take two line L1 and L2
    Output: the intersection between the two lines
    '''
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
    
def get_core_triangle(kpCenterBbox):

    '''
    Input: all the point normalised to the bbox (panda Dataframe)
    Output: position of the core 
    
    '''
    p5 = kpCenterBbox.loc[5][["x","y"]].to_list()
    p12 = kpCenterBbox.loc[12][["x","y"]].to_list()
    p11 = kpCenterBbox.loc[11][["x","y"]].to_list()
    p6 = kpCenterBbox.loc[6][["x","y"]].to_list()
    L1 = get_line(p5, p12)
    L2 = get_line(p6, p11)
    inter= get_intersection(L1, L2)
    return inter

#------------------------------------------------------------------------------------

def align_student_to_teacher(bodyPart_T, bodyPart_S):

    '''
    Input: take a bodypart of two poses (e.g. teacher and student)
    Output: align with a rigid transform the bodypart of the student to the bodypart of the teacher (bodyPart_sPrime)
    '''

    bodyPart_T["visibility"]= bodyPart_T.confidence>0.5
    bodyPart_S["visibility"]= bodyPart_S.confidence>0.5
    indexDetect_T=bodyPart_T[bodyPart_T.visibility==True].index
    indexDetect_S=bodyPart_S[bodyPart_S.visibility==True].index
    shared_ind=indexDetect_T[indexDetect_T.isin(indexDetect_S)]
    if shared_ind.shape[0] >=2:
        teacher=np.array(bodyPart_T.loc[shared_ind][["x","y"]],dtype=np.float32)
        student=np.array(bodyPart_S.loc[shared_ind][["x","y"]],dtype=np.float32)
        m=cv2.estimateRigidTransform(student, teacher,True)
        sPrime=np.dot(student, m[:,:2].T)+ m[np.newaxis,:,2]
        bodyPart_sPrime = pd.DataFrame({"x": sPrime[:,0], "y": sPrime[:,1], "confidence":bodyPart_S.loc[shared_ind].confidence })
        return bodyPart_sPrime

#------------------------------------------------------------------------------------

def align_core_to_center(kpCenterBbox_T,dst=(0.5,0.5)):

    '''
    Input: part of the body and the destinationpoint
    Output: skeleton core align on the center of the bbox
    '''

    core_T=get_core_triangle(kpCenterBbox_T)
    dx_T,dy_T=core_T[0]-dst[0],core_T[1]-dst[1]
    kpTp=kpCenterBbox_T.copy()
    kpTp["x"],kpTp["y"]=kpCenterBbox_T.x-dx_T,kpCenterBbox_T.y-dy_T
    
    return kpTp

#------------------------------------------------------------------------------------

def translate_bodyPart(torso_S,torso_Sp,pFHead_S,pFArmLeft_S,pFArmRight_S,pFTorso_S,pFLegLeft_S,pFLegRight_S):

    '''
    Input: all body part of the same way that the torso have been translated 
    Output: all modified dataframe ready to be reconstructed 
    '''

    topL_dx,topL_dy=torso_Sp.loc[6].x-torso_S.loc[6].x , torso_Sp.loc[6].y-torso_S.loc[6].y
    topR_dx,topR_dy=torso_Sp.loc[5].x-torso_S.loc[5].x , torso_Sp.loc[5].y-torso_S.loc[5].y
    botL_dx,topL_dy=torso_Sp.loc[12].x-torso_S.loc[12].x , torso_Sp.loc[12].y-torso_S.loc[12].y
    botR_dx,topR_dy=torso_Sp.loc[11].x-torso_S.loc[11].x , torso_Sp.loc[11].y-torso_S.loc[11].y
    pFHead_S_right=pFHead_S.loc[[2,4]]
    pFHead_S_Left=pFHead_S.loc[[1,3]]
    
    pFHead_S_right["x"],pFHead_S_right["y"]=pFHead_S_right.x+topR_dx,pFHead_S_right.y+topR_dy
    pFArmRight_S["x"],pFArmRight_S["y"]= pFArmRight_S.x+topR_dx,pFArmRight_S.y+topR_dy
    pFLegRight_S["x"],pFLegRight_S["y"]=pFLegRight_S.x+botR_dx,pFLegRight_S.y+topR_dy
    
    pFHead_S_Left["x"],pFHead_S_Left["y"]=pFHead_S_Left.x+topL_dx,pFHead_S_Left.y+topL_dy
    pFArmLeft_S["x"],pFArmLeft_S["y"]= pFArmLeft_S.x+topL_dx,pFArmLeft_S.y+topL_dy
    pFLegLeft_S["x"],pFLegLeft_S["y"]=pFLegLeft_S.x+botL_dx,pFLegLeft_S.y+topL_dy
    
    pFArmLeft_Sp,pFArmRight_Sp,pFLegLeft_Sp,pFLegRight_Sp= pFArmLeft_S,pFArmRight_S,pFLegLeft_S,pFLegRight_S
    pFHead_Sp=pd.concat([pFHead_S_right,pFHead_S_Left])
    return pFHead_Sp,pFArmLeft_Sp,pFArmRight_Sp,pFLegLeft_Sp,pFLegRight_Sp

#------------------------------------------------------------------------------------

def reconstruction_Sp(pFHead_Sp,pFArmLeft_Sp,pFArmRight_Sp,pFTorso_Sp,pFLegLeft_Sp,pFLegRight_Sp):

    '''
    Input: all modified bodypart 
    Output: reconstruction by taking the mean of the position
    '''

    kpCenterBbox_Sp=pd.concat([pFTorso_Sp,pFHead_Sp,pFArmLeft_Sp,pFArmRight_Sp,pFTorso_Sp,pFLegLeft_Sp,pFLegRight_Sp],axis=0)
    reconstructed_df=kpCenterBbox_Sp.reset_index().groupby(["index"]).mean()
    return reconstructed_df


#------------------------------------------------------------------------------------

def get_jpss(kpCenterBbox_T,kpCenterBbox_Sp,with_conf):

    '''
    Input: two pandadataframe : kpCenterBbox_T ( contains all points of the teacher ) and kpCenterBbox_Sp (contains all reconstructed keypoints)
    Output: Joint Position similatity evaluation
    
    '''
    if with_conf==False:
        #boolean if the point is find
        mask_T = kpCenterBbox_T["confidence"].notna()
        mask_Sp =kpCenterBbox_Sp["confidence"].notna()
        #apply 1 of confidence to all the point which have been found
        kpCenterBbox_T["confidence"][mask_T]=1
        kpCenterBbox_Sp["confidence"][mask_Sp]=1
 
    # take only the point with good confidence  in case we are dealing with confidence
    kpCenterBbox_T["visibility"]= kpCenterBbox_T.confidence>0.5
    kpCenterBbox_Sp["visibility"]= kpCenterBbox_Sp.confidence>0.5

    #find the index that are both visible 
    indexDetect_T=kpCenterBbox_T[kpCenterBbox_T.visibility==True].index
    indexDetect_Sp=kpCenterBbox_Sp[kpCenterBbox_Sp.visibility==True].index

    shared_ind=indexDetect_T[indexDetect_T.isin(indexDetect_Sp)]
    shared_kp_T=kpCenterBbox_T.loc[shared_ind]
    shared_kp_Sp=kpCenterBbox_Sp.loc[shared_ind]

    #flatten 
    t_flat=np.array(shared_kp_T[["x","y"]]).flatten()
    sp_flat=np.array(shared_kp_Sp[["x","y"]]).flatten()

    #scale_calculation 
    x_max=np.maximum(np.array(shared_kp_T.x),np.array(shared_kp_Sp.x)).max()
    x_min=np.minimum(np.array(shared_kp_T.x),np.array(shared_kp_Sp.x)).min()
    y_max=np.maximum(np.array(shared_kp_T.y),np.array(shared_kp_Sp.y)).max()
    y_min=np.minimum(np.array(shared_kp_T.y),np.array(shared_kp_Sp.y)).min()

    # square root of the square root of the object segment area
    s=np.sqrt((x_max-x_min)*(y_max-y_min))

    # calcul the error (MSE)
    e=t_flat-sp_flat
    error_joint=np.sqrt(np.sum(e.reshape(shared_kp_T[["x","y"]].shape)**2,axis=1))
    #create a df similar to shared_kp_sp and add the error
    oks=shared_kp_Sp.copy().reset_index()
    oks["distanceT_Sprime"]=error_joint

    # and the confidence 
    oks["T_conf"]=shared_kp_T.confidence
    oks["Sp_conf"]=shared_kp_Sp.confidence
    oks["confidence"]=oks[["T_conf","Sp_conf"]].mean(axis=1)
    

    # the constant have been find at this url :
    # https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78#:~:text=Object%20Keypoint%20Similarity%20%28OKS%29%20%E2%80%9CIt%20is%20calculated%20from,location%20more%20precise%20than%20hip%20location.%E2%80%9D%20%E2%80%94%20http%3A%2F%2Fcocodataset.org%2F?msclkid=fd487b70ba3d11ec85bc452a6360e8ef
    
    k=np.array([0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089])
    oks["k"]=k[shared_ind]
    oks=oks.set_index("index")
    # if not we set the conf_type to none oks.confidence will be only one
    return np.nansum(np.exp(-oks["distanceT_Sprime"]**2/(2*s**2*oks.k**2))*oks.confidence)/np.nansum(oks.confidence)

#------------------------------------------------------------------------------------
        
def get_jass(kpCenterBbox_T,kpCenterBbox_S,conf_type,thresholdAngle,type_norm):

    '''
    Input: two panda dataframe : kpCenterBbox_T ( contains all points of the teacher ) and kpCenterBbox_S (contains keypoints position before reconstruction)
    Output: Joint anglular similarity evaluation
    
    '''
    
    angleTeacher=hp.create_df_features_angles(kpCenterBbox_T,type_norm,conf_type)
    angleStudent=hp.create_df_features_angles(kpCenterBbox_S,type_norm,conf_type)
    conf_df=pd.concat([angleTeacher,angleStudent]).loc["confidence"]
    angleDiff=(angleTeacher.T.metric-angleStudent.T.metric).reset_index()

    if conf_type=="mean":
        angleDiff["confidence"]=np.mean(conf_df.values,axis=0)
        
    elif conf_type=="min":
        angleDiff["confidence"]=np.min(conf_df.values,axis=0)
    
    angleDiff=angleDiff.set_index("index")
    #jass = 1 if the angle are the same 0 if over the threshold else we use the equation presented in the report
    angleDiff.loc[np.abs(angleDiff.metric)==0,"metric"]=1
    angleDiff.loc[np.abs(angleDiff.metric)>thresholdAngle,"metric"]=0
    angleDiff.loc[(np.abs(angleDiff.metric)<=thresholdAngle)&(np.abs(angleDiff.metric)>0),"metric"]=np.sqrt(((thresholdAngle**2)-(angleDiff.metric[(np.abs(angleDiff.metric)<=thresholdAngle)&(np.abs(angleDiff.metric)>0)]**2)).astype("float64"))/thresholdAngle
    jass=angleDiff.metric
    if conf_type!="none":
        jass=jass/angleDiff.confidence
    return jass.mean()

#------------------------------------------------------------------------------------

def get_pas(FILE_S,FILE_T,type_norm,sub_method,thresholdAngle,normalisation,conf_type,with_conf,ratio_pas):

    '''
    Input: student and teacher file path (FILE_S,FILE_T), ratio used between jass and jpss (ratio_pas), norm used to calculate the angles and segments (type_norm),
    , the submethod used align student/teacher, the treshold angle for the jass, a flag to see the plot or just the result,
    , the type of normalisation and the type of confidence treatment.
    Output: Joint anglular similarity evaluation
    '''

    _,_,_,pFTorso_T,_,_,bbox_T,kpCenterBbox_T=separate_body_part(FILE_T,normalisation)
    pFHead_S,pFArmLeft_S,pFArmRight_S,pFTorso_S,pFLegLeft_S,pFLegRight_S,bbox_S,kpCenterBbox_S=separate_body_part(FILE_S,normalisation)

    if sub_method=="align_torso":
        pFTorso_Sp=align_student_to_teacher(pFTorso_T, pFTorso_S)
        pFHead_Sp,pFArmLeft_Sp,pFArmRight_Sp,pFLegLeft_Sp,pFLegRight_Sp=translate_bodyPart(pFTorso_S,pFTorso_Sp,pFHead_S,pFArmLeft_S,pFArmRight_S,pFTorso_S,pFLegLeft_S,pFLegRight_S)
        kpCenterBbox_Sp=reconstruction_Sp(pFHead_Sp,pFArmLeft_Sp,pFArmRight_Sp,pFTorso_Sp,pFLegLeft_Sp,pFLegRight_Sp)
        kpTp,kpSp=kpCenterBbox_T,kpCenterBbox_Sp
        
    if sub_method=="core":
        kpTp=align_core_to_center(kpCenterBbox_T,dst=(0.5,0.5))
        kpSp=align_core_to_center(kpCenterBbox_S,dst=(0.5,0.5))

    # calculation 
    jpss=get_jpss(kpTp,kpSp,with_conf)


    jass=get_jass(kpCenterBbox_T,kpCenterBbox_S,conf_type,thresholdAngle,type_norm)

    return (jass*ratio_pas)+jpss*ratio_pas
