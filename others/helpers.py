# -*- coding: utf-8 -*-
from msilib.schema import File
import numpy as np 
import pandas as pd
import others.pas as pas
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import others.visualisation as kp_visu


# Hard coded segments and angle for a human output of OpenPifPaf
segments_human = np.array([[3, 5], [4, 6], [5, 7], [6, 8], [7,9], [8, 10], [5, 6], [5, 11], [6, 12],[11, 12], [11, 13], [12, 14], [13, 15], [14, 16]])
angles_human = np.array([[3,5,6], [4, 6,5], [6, 5,11], [5, 6,12], [7,5,11], [8,6,12], [5, 7,9], [6,8,10], [5,11,12],[6,12,11], [13,11,12], [14,12,11], [15,13,11], [16,14,12]])

#helpers

# normalisation 
def MinMaxNorm(data):

    '''
    Normalise the data input
    '''

    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalise(point_df,bbox,type):

    ''' 
    Input: take the keypoints with np.nan when point are not detected (point_df) [Dataframe]
    Output: normalise the keypoint between 0 and 1 on the x axis
    '''
    
    top_left_x, top_left_y, w, h = bbox
    max_axis_x = top_left_x+w 
    max_axis_y = top_left_y+h
    min_axis_x = top_left_x
    min_axis_y = top_left_y
    #normalise the keypoint base on the bbox
    keypoints_normalise_x = (point_df.loc[:,['x']]-min_axis_x)/(max_axis_x-min_axis_x)
    keypoints_normalise_y = (point_df.loc[:,['y']]-min_axis_y)/(max_axis_y-min_axis_y)
    ratio_hw= h/w
    
    if type == "bbox_ratio_kept":
        keypoints_normalise = pd.concat([keypoints_normalise_x,ratio_hw*keypoints_normalise_y, point_df.confidence],axis=1)
        return keypoints_normalise
    
    if type == "bbox_ratio_kept_center_core":
        keypoints_normalise = pd.concat([keypoints_normalise_x,ratio_hw*keypoints_normalise_y, point_df.confidence],axis=1)
        return pas.align_core_to_center(keypoints_normalise,dst=(0.5,0.5))
    
    if type == "none":
        return point_df

#------------------------------------------------------------------------------------

#retrieval from JPG

def get_pose_fromJson_JPG(self):

    ''' 
    Input: Json and which frame we want to extract
    Output: dataframe with nan where the confidence is 0 and reshaped
    '''

    #get if they have a subject detected
    df = pd.read_json(self)
    df = np.array(df)[0][0]
    kp = df.get("keypoints")
    keypoints = np.asarray(kp).reshape(17,3)
    bbox = np.asarray(df.get("bbox"))
    point = pd.DataFrame(keypoints,columns=["x","y","confidence"])
    #deal with the point which have 0 confidence
    point[point.confidence==0] = np.nan
    

    return point,bbox

#------------------------------------------------------------------------------------

#Calcul the distance between the points and calcul the angle 

def cal_joint_len(point_a,point_b,point_df,type_norm):

    '''
    Input: two points [int], the type of norm [int] and the point datasets [Dataframe]
    Output: Mesure the distance btw two point
    '''

    if (point_df.iloc[point_a,1]<0)|(point_df.iloc[point_b,1]<0) : 
        return np.nan
    else:
        return np.linalg.norm(point_df.iloc[point_a,0:2]-point_df.iloc[point_b,0:2],ord=type_norm)
    
def cal_angle(point_a,point_b,point_c,points_df,type_norm):

    '''
    Input: three points  [int], the type of the norm [int] and the points datasets [Dataframe]
    Output: Mesure the angle btw the three points in radian and return the quaternion of the angle
    '''
    if(points_df.iloc[point_a,1]<0)|(points_df.iloc[point_b,1]<0)|(points_df.iloc[point_c,1]<0) : 
        return np.nan
    else: 
        ba = points_df.iloc[point_a,0:2] - points_df.iloc[point_b,0:2]
        bc = points_df.iloc[point_c,0:2] - points_df.iloc[point_b,0:2]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba,ord=2) * np.linalg.norm(bc,ord=type_norm))
    
        return np.arccos(cosine_angle)*180/np.pi #Quaternion(axis=[0, 0, 1], angle=np.arccos(cosine_angle))

#------------------------------------------------------------------------------------

# dealing with confidence

def mean_confidence_j(points,point_a,point_b,conf_type):

    '''
    Input: 2 point [int], the dataset [Dataframe] and the type of confidence propagation [string]
    Output: the confidence for a joint
    '''

    if (points.iloc[point_a,2]==np.nan)|(points.iloc[point_b,2]==np.nan) : 
        return np.nan
    else:
        if conf_type == "mean":
            return (np.sum([points.iloc[point_a,2],points.iloc[point_b,2]]))/2
        if conf_type == "min":
            return np.min([points.iloc[point_a,2],points.iloc[point_b,2]])
        if conf_type == "none":
            return np.ones_like(points.iloc[point_a,2])


#------------------------------------------------------------------------------------

def mean_confidence_a(points,point_a,point_b,point_c,conf_type):

    '''
    Input: 3 points [int], the dataset [Dataframe] and the type of confidence propagation [string]
    Output: the confidence for a angle
    
    '''
    
    if(points.iloc[point_a,2]==np.nan)|(points.iloc[point_b,2]==np.nan)|(points.iloc[point_c,2]==np.nan) : 
        return np.nan
    else:
        if conf_type == "mean":
            return (np.sum([points.iloc[point_a,2],points.iloc[point_b,2],points.iloc[point_c,2]]))/3
        if conf_type == "min":
            return np.min([points.iloc[point_a,2],points.iloc[point_b,2],points.iloc[point_c,2]])
        if conf_type == "none":
            return np.ones_like(points.iloc[point_a,2])

#------------------------------------------------------------------------------------

#create the features    
def create_df_features(point_df,conf_type,type_norm,points,angles,segments):
    '''
    Input: point_dataset [Dataframe], the type of confidence propagation [string], the type of norm [int] and different flag 
    if we are taking into account the points,angles and segments  [bool]
    Output: all created features for a pose
    '''

    if points == True:

        px = ["p"+str(i)+"_x" for i in range(point_df.shape[0])]
        py = ["p"+str(i)+"_y" for i in range(point_df.shape[0])]
        p_x = point_df[["x","confidence"]]
        
        if conf_type == "none":
            p_x.loc[:,"confidence"]=1.0

        p_x['p'] = px
        p_x = p_x.set_index('p')
        # normalisation for x
        p_x["x"] = MinMaxNorm(p_x["x"])
        p_y = point_df[["y","confidence"]]

        if conf_type == "none":
            p_y.loc[:,"confidence"]=1.0

        p_y['p'] = py
        p_y = p_y.set_index('p')
        # normalisation for y
        p_y["y"] = MinMaxNorm(p_y["y"])

        if (angles == True) and (segments == True) :

            name_list_j = ['j_'+str(x+1) for x in range(14)]
            name_list_a = ['a_'+str(x+1) for x in range(14)]

            
            len_list_a = [cal_joint_len(p1,p2,point_df,type_norm) for [p1,p2] in segments_human]
            len_list_j = [cal_angle(a1,a2,a3,point_df,type_norm) for [a1,a2,a3] in angles_human]
        
            confidence_list_j = [mean_confidence_j(point_df,p1,p2,conf_type) for [p1,p2] in segments_human]
            confidence_list_a = [mean_confidence_a(point_df,a1,a2,a3,conf_type) for [a1,a2,a3] in angles_human]
            
            df_a = pd.DataFrame([len_list_a,confidence_list_a],columns=name_list_a)
            df_j = pd.DataFrame([len_list_j,confidence_list_j],columns=name_list_j)

            #normalisation for joints/segments and angle
            df_a.iloc[0,:] = MinMaxNorm(df_a.iloc[0,:])
            df_j.iloc[0,:] = MinMaxNorm(df_j.iloc[0,:])

            df = pd.concat([df_a.reset_index(drop=True),df_j.reset_index(drop=True),p_x.T.reset_index(drop=True),p_y.T.reset_index(drop=True)],axis=1)
            df["index"] = ['metric','confidence']

            return df.set_index("index")
        
        if (angles == False) and (segments == True) :

            name_list = ['j_'+str(x+1) for x in range(14)]
            len_list = [cal_joint_len(p1,p2,point_df,type_norm) for [p1,p2] in segments_human]
            confidence_list = [mean_confidence_j(point_df,p1,p2,conf_type) for [p1,p2] in segments_human]

            df_j = pd.DataFrame([len_list,confidence_list],columns=name_list)

            #normalisation segments
            df_j.iloc[0,:] = MinMaxNorm(df_j.iloc[0,:])

            df = pd.concat([df_j.reset_index(drop=True),p_x.T.reset_index(drop=True),p_y.T.reset_index(drop=True)],axis=1)
            df["index"] = ['metric','confidence']

            return df.set_index("index")
        
        if (segments == False) and (angles == True) :

            name_list = ['a_'+str(x+1) for x in range(14)]
            len_list = [cal_angle(a1,a2,a3,point_df,type_norm) for [a1,a2,a3] in angles_human]
            confidence_list = [mean_confidence_a(point_df,a1,a2,a3,conf_type) for [a1,a2,a3] in angles_human]

            df_a=pd.DataFrame([len_list,confidence_list],columns=name_list)

            #normalisation angle
            df_a.iloc[0,:]=MinMaxNorm(df_a.iloc[0,:])

            df=pd.concat([df_a.reset_index(drop=True),p_x.T.reset_index(drop=True),p_y.T.reset_index(drop=True)],axis=1)
            df["index"]=['metric','confidence']

            return df.set_index("index")
        
        if  (segments == False) and (angles == False):

            df = pd.concat([p_x.T.reset_index(drop=True),p_y.T.reset_index(drop=True)],axis=1)
            df["index"] = ['metric','confidence']

            return df.set_index("index")
                     
    else: 
        
        if (angles == True) and (segments == True) :

            name_list_j = ['j_'+str(x+1) for x in range(14)]
            name_list_a = ['a_'+str(x+1) for x in range(14)]
            
            len_list_a = [cal_joint_len(p1,p2,point_df,type_norm) for [p1,p2] in segments_human]
            len_list_j = [cal_angle(a1,a2,a3,point_df,type_norm) for [a1,a2,a3] in angles_human]
        
            confidence_list_j = [mean_confidence_j(point_df,p1,p2,conf_type) for [p1,p2] in segments_human]
            confidence_list_a = [mean_confidence_a(point_df,a1,a2,a3,conf_type) for [a1,a2,a3] in angles_human]
            
            df_a = pd.DataFrame([len_list_a,confidence_list_a],columns=name_list_a)
            df_j = pd.DataFrame([len_list_j,confidence_list_j],columns=name_list_j)

            #normalisation
            df_a.iloc[0,:] = MinMaxNorm(df_a.iloc[0,:])
            df_j.iloc[0,:] = MinMaxNorm(df_j.iloc[0,:])

            df = pd.concat([df_a.reset_index(drop=True),df_j.reset_index(drop=True)],axis=1)
            df["index"] = ['metric','confidence']

            return df.set_index("index")
        
        if (angles == False) and (segments == True) :

            name_list = ['j_'+str(x+1) for x in range(14)]
            len_list = [cal_joint_len(p1,p2,point_df,type_norm) for [p1,p2] in segments_human]
            
            confidence_list = [mean_confidence_j(point_df,p1,p2,conf_type) for [p1,p2] in segments_human]
            df_j = pd.DataFrame([len_list,confidence_list],columns=name_list)

            #normalisation for segments
            df_j.iloc[0,:] = MinMaxNorm(df_j.iloc[0,:])

            df_j["index"]=['metric','confidence']

            return df_j.set_index("index")
        
        if (segments == False) and (angles == True):

            name_list = ['a_'+str(x+1) for x in range(14)]
            len_list = [cal_angle(a1,a2,a3,point_df,type_norm) for [a1,a2,a3] in angles_human]
            confidence_list = [mean_confidence_a(point_df,a1,a2,a3,conf_type) for [a1,a2,a3] in angles_human]

            df_a = pd.DataFrame([len_list,confidence_list],columns=name_list)

            #normalisation angle
            df_a.iloc[0,:] = MinMaxNorm(df_a.iloc[0,:])
            df_a["index"]=['metric','confidence']

            return df_a.set_index("index")

#------------------------------------------------------------------------------------

# Calcul du loss
def calculate_dist_eucl(x1,x2,c1,c2,loss_conf_type,nan):
    
    '''
    Input: vector x1,x2 with their confidence c1 c2 [numpy array], method used to take care of the confidence(conf_type) [string], how the nan should be treated (nan) [string]
    (If nan=mean we will not take into account the nan and therefore a mean of the distance is returned )
    Output: error between the two features vectors as euclidean distance
    '''

    #propagation of the confidence

    if loss_conf_type == "mean":
        c=(c1+c2)/2
        
    if loss_conf_type == "min":
        c=np.min([c1,c2],axis=0)
        
    if loss_conf_type == "none":
        c=np.ones_like(c1)
        
    e = np.array(x1)-np.array(x2)

    # dealing with the NaN
    if nan == "zero":
        e = np.nan_to_num(e, nan=0)
        dist = np.sqrt(np.sum(e**2))
        return np.sum(dist)/np.sum(~pd.isnull(x1-x2))
    
    if nan == "mean":
        e = e/c
        dist = np.sqrt(np.nansum(e**2))
        count_not_nan = np.sum(~pd.isnull(x1-x2))
        
        if count_not_nan == 0:
            print("Error: We only have NaN")
            return np.inf
        else:
            return np.sum(dist/count_not_nan)

#------------------------------------------------------------------------------------

#calcul the loss by takin into account the confidence 
def calculate_loss_with_confidence(pose1,pose2,loss_type,loss_conf_type):

    '''
    Input: 2 poses [Dataframe] and the type of the loss (loss_type) [string],
    and how to propagates the conf in the when computing the loss (loss_conf_type)[string],
    Outuput: loss by taking into account the confidence
    '''

    x1,c1,x2,c2 = pose1.T.metric,pose1.T.confidence,pose2.T.metric,pose2.T.confidence

    if loss_type == "euclidean_distance_mean":
        return calculate_dist_eucl(x1,x2,c1,c2,loss_conf_type,nan="mean")
    if loss_type == "euclidean_distance_zero":
        return calculate_dist_eucl(x1,x2,c1,c2,loss_conf_type,nan="zero")

#------------------------------------------------------------------------------------

def create_df_features_points(point_df,conf_type):

    '''
    Input: keypoints dataset normalised [Dataframe] and how to propagates conf [string]
    Output: keypoint feature with both coordonnee flatten on 1 vector
    '''

    px = ["p"+str(i)+"_x" for i in range(point_df.shape[0])]
    py = ["p"+str(i)+"_y" for i in range(point_df.shape[0])]
    p_x = point_df[["x","confidence"]]

    if conf_type == "none":
            p_x.loc[:,"confidence"]=1.0

    p_x['p'] = px
    p_x = p_x.set_index('p')
    p_y = point_df[["y","confidence"]]

    if conf_type == "none":
            p_x.loc[:,"confidence"]=1.0

    p_y['p'] = py
    p_y = p_y.set_index('p')
    df_ = pd.concat([p_x.T.reset_index(drop=True),p_y.T.reset_index(drop=True)],axis=1)
    df_["index"] = ['metric','confidence']
    return df_.set_index("index")

#------------------------------------------------------------------------------------

def create_df_features_angles(point_df,type_norm,conf_type):

    '''
    Input: keypoints  dataset normalised [Dataframe], type of the norm [int], how to deal with the conf [string]
    Output: keypoint features for all angles of the body flatten on one vector
    '''
    
    name_list=['a_'+str(x+1) for x in range(14)]
    len_list=[cal_angle(a1,a2,a3,point_df,type_norm) for [a1,a2,a3] in angles_human]
    confidence_list=[mean_confidence_a(point_df,a1,a2,a3,conf_type) for [a1,a2,a3] in angles_human]
    df_a=pd.DataFrame([len_list,confidence_list],columns=name_list)
    df_=pd.concat([df_a.reset_index(drop=True)],axis=1)
    df_["index"]=['metric','confidence']
    return df_.set_index("index")

#------------------------------------------------------------------------------------

def create_df_features_segments(point_df,type_norm,conf_type):

    '''
    Input: keypoints  dataset normalised [Dataframe], type of the norm [int], how to deal with the conf [string]
    Output: keypoint feature all segments of the body flatten on one vector
    '''

    name_list = ['j_'+str(x+1) for x in range(14)]
    len_list = [cal_joint_len(p1,p2,point_df,type_norm) for [p1,p2] in segments_human]
    confidence_list = [mean_confidence_j(point_df,p1,p2,conf_type) for [p1,p2] in segments_human]

    df_j = pd.DataFrame([len_list,confidence_list],columns=name_list)
    df_ = pd.concat([df_j.reset_index(drop=True)],axis=1)
    df_["index"] = ['metric','confidence']

    return df_.set_index("index")

    # ----------------------------------------------------------------------------------------------------------------------------------
    # helpers for video 

def get_features_for_frame(frame_pose,nb_min_points_detected):

    '''
    Input : frame pose from a video [DataFrame], threshold on the detected points (nb_min_points_detected) [int]
    Output : features and confidence based on the best combination find in our analysis by images
    
    '''

    #best combination
    normalisation="bbox_ratio_kept_center_core"
    conf_type="mean"
    type_norm=2
    points=True
    angles=True
    segments=False

    #computation
    points_df = pd.DataFrame(np.array(frame_pose["keypoints"]),columns=["x","y","confidence"])
    points_df[points_df.confidence == 0] = np.nan
    if points_df.confidence.notna().sum()>nb_min_points_detected:
        pt_normalise = normalise(points_df,np.array(frame_pose["bbox"]),normalisation)
        feature_vector = create_df_features(pt_normalise,conf_type,type_norm,points,angles,segments)
        features = feature_vector.head(1)
        confidences = feature_vector.tail(1)
        return features,confidences
    else:
         return (None,None)
    

def video_dimensional_analysis(FILE,nb_min_points_detected,clust=True,clust_type="kmeans",visu=False):
    '''
    INPUT: 
    adress of the file which is compared with the other (FILE) [string],
    threshold on the number of point detected (nb_min_points_detected) [int],
    flag in order to say if we want a clusting on the data or not (clust) [bool],
    define the type of cluster (clust_type) [string],
    flag if we want to visualise or not the four chosen images and the embeddings (visu) [bool]

    OUTPUT: 
    groups of the features for all for images unstacked and the same for the confidence
    '''

    print(FILE)

    #Extract Data
    df=pd.read_json(FILE, lines=True)
    mask_nonempty=df.predictions.apply(lambda x: len(x)!=0)
    df=df[mask_nonempty].reset_index().drop("index",axis=1)
    df["keypoints"]=df.predictions.apply(lambda x: np.array(x[0].get("keypoints")).reshape(17,3))
    df["bbox"]=df.predictions.apply(lambda x:np.asarray(x[0].get("bbox")))
    df["score"]=df.predictions.apply(lambda x:np.asarray(x[0].get("score")))
    df["id"]=df.predictions.apply(lambda x:np.asarray(x[0].get("category_id")))
    df=df.drop("predictions",axis=1)

    # Sorting on the score of the pose ( doesn't want a video that only have scores under 0.6) 
    groups=pd.DataFrame()

    if df.score.max() >= 0.6:

        # select only the subject of interest of the video
        stat_frames=np.unique(df.id,return_counts=True)
        IOI=stat_frames[1].argmax() # id of interest we taje the id the most represented in the video
        df_SOI=df[df.id==stat_frames[0][IOI]]
        df_SOI=df_SOI.reset_index(drop=True)

        df_SOI["features"]=df_SOI.apply(lambda x: get_features_for_frame(x,nb_min_points_detected),axis=1)
        metric=pd.DataFrame(df_SOI["features"].tolist()).rename({0:"features",1:"confidences"},axis=1)

        #compute the Dataframes with all features and all the confidence
        features_all=np.array(metric.features)
        confidences_all=np.array(metric.confidences)
        features_all,confidences_all=pd.concat(features_all).reset_index(drop=True),pd.concat(features_all).reset_index(drop=True)

        frames_del=df.shape[0]-features_all.shape[0]
        print("Nombre de frame not folowing the minimum number detected : "+str(frames_del))

        
        # separate the video in four
        ind_time_seg=np.floor(np.linspace(0,features_all.shape[0]-1-frames_del,4)).astype(int)

        # We treat the na with the mean on the lines to be able to reduct the dimension
        features_all_embedding=features_all.fillna(features_all.mean(axis=0)) # add mean inplace of nan
        #confidences_all=confidences_all.fillna(0) # fill by 0 if not detected
        features_all_embedding=features_all.fillna(0) # fill by 0 if not detected
        X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(features_all_embedding) # transform en 2D

        if visu==True:
            #plot the embedding in two dimension
            fig, axs = plt.subplots(1, 2, figsize=(14,6), sharey=True)
            axs[0].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], alpha=0.6)
            axs[0].set_title("TSNE",fontweight="bold")

        if clust==True: 
            #apply an embedding on the data and then separate into 4 cluster
            if clust_type=="kmeans":
                dba_km = TimeSeriesKMeans(n_clusters=4,
                            n_init=2,
                            metric='euclidean',
                            max_iter_barycenter=10,
                            random_state=10).fit(features_all_embedding)
                labels=dba_km.labels_


            if visu==True:
                # Plot the data reduced in 2d space with TSNE with cluster
                axs[1].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=labels, alpha=0.6)
                axs[1].legend(np.unique(labels))
                axs[1].set_title("TSNE-"+clust_type,fontweight="bold")
                plt.xticks(fontweight="bold")
                plt.yticks(fontweight="bold")
                plt.show()

            # Add label and frame
            features_all["label"]=labels
            confidences_all["label"]=labels

            #add the frame in order to plot 
            features_all["frame"]=df_SOI.copy()["frame"]
            confidences_all["frame"]=features_all["frame"]

            #choose randomly a point into those cluster
            np.random.seed(3)
            groups=features_all.groupby(["label"]).sample(1)
            groups_conf=confidences_all.loc[groups.index.values,:]

            #drops the label in both groups
            groups=groups.drop("label",axis=1)
            groups_conf=groups_conf.drop("label",axis=1)
            
            
        elif clust==False:
            if clust_type == "minMax_xy":
                # take four sample of our embedding based on their localisation. Those will represent the most different frames in the space 
                # ( further left,right,up and down)
                x_ind_max=pd.DataFrame(X_reduced_tsne)[0].argmax()
                y_ind_max=pd.DataFrame(X_reduced_tsne)[1].argmax()
                x_ind_min=pd.DataFrame(X_reduced_tsne)[0].argmin()
                y_ind_min=pd.DataFrame(X_reduced_tsne)[1].argmin()
                if visu==True:
                    # plot the data reduced in 2d with red points at which point were used
                    axs[1].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], alpha=0.6)
                    axs[1].scatter(X_reduced_tsne[[x_ind_max,x_ind_min,y_ind_max,y_ind_min]][:,0], X_reduced_tsne[[x_ind_max,x_ind_min,y_ind_max,y_ind_min]][:,1],c="red" ,alpha=0.6)
                    axs[1].set_title("Most different frame exploration",fontweight="bold")
                    plt.xticks(fontweight="bold")
                    plt.yticks(fontweight="bold")
                    plt.show()

                # Add label and frame
                features_all["frame"]=df_SOI.copy()["frame"]
                confidences_all["frame"]=df_SOI.copy()["frame"]

                #form the groups
                groups=features_all.iloc[[x_ind_max,x_ind_min,y_ind_max,y_ind_min],:]
                groups_conf=confidences_all.iloc[[x_ind_max,x_ind_min,y_ind_max,y_ind_min],:]

            elif clust_type == "time_segmentation": 
                # use as four sample the four separation calculated before 
                # plot the data reduced in 2d
                if visu==True:        
                    #plot those points on the embedding
                    axs[1].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], alpha=0.6)
                    axs[1].scatter(X_reduced_tsne[ind_time_seg][:,0], X_reduced_tsne[ind_time_seg][:,1],c="red" ,alpha=0.6)
                    axs[1].set_title("Time segmentation frame exploration",fontweight="bold")
                    plt.xticks(fontweight="bold")
                    plt.yticks(fontweight="bold")
                    plt.show()

                # Add label and frame
                features_all["frame"]=df_SOI.copy()["frame"]
                confidences_all["frame"]=df_SOI.copy()["frame"]

                #form the groups
                groups=features_all.iloc[ind_time_seg,:]
                groups_conf=confidences_all.iloc[ind_time_seg,:]
                

        if visu==True:
            # plot the image that will be used 
            plt.subplots(1, groups.shape[0], figsize=(14,6))
            for sp,i in zip(range(1,groups.shape[0]+1),list(groups.frame)):
                kp_visu.print_frame(FILE,i,sp,groups.shape[0])
            plt.show()
            
        #drop the frame in all groups
        groups=groups.drop("frame",axis=1)
        groups_conf=groups_conf.drop("frame",axis=1)
        print("Features")
        print(groups)
        print("Confidence")
        print(groups_conf)
    return groups.unstack().values,groups_conf.unstack().values


