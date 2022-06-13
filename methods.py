
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,recall_score
from annoy import AnnoyIndex
from os import listdir
pd.options.mode.chained_assignment = None  # default='warn'
#visu
import matplotlib.pyplot as plt 
import cv2
# other files
import others.pas  as pas
import others.helpers as hp
import time
import random 
import openpifpaf


# -------------------------------------------------------------------------------------------------------------------------------------------
def knn_modified(LABEL_LIST,ALL_FILES,loss_type,k,normalisation,type_norm,points,angles,segments,conf_type,with_conf,loss_conf_type,visu=True):

    ''' 
    INPUT: 
    This function takes as argument a list of the unique label (LABEL_LIST) [string], the list with all the files that we want to compare (ALL_FILES)[string],
    the type of loss used to enhance the neighbors (loss_type)[string], the number of neighbors voting to choose the label (k)[int], the type of normalisation 
    used on the skeletton keypoints (normalisation) [string], the type of norm used to calculate angles and segments (type_norm)[int],
    flags if we take in consideration points,angles or segments [bool] and the choice of the method to treat the confidence (conf_type)[bool]
        
    Output: 
    Recall matrix of the comparison between all poses and the confusion matrix for the different classes
    '''
    
    # Start timer 
    start = time.time() 

    # Initialisation
    poses = []
    lab_all_files = []
    pt_df,ang_df,seg_df = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    # create for each files the wanted features
    for FILE in ALL_FILES:

        #1 get the point and the Json from the file 
        pt,bbox = hp.get_pose_fromJson_JPG(FILE)

        #2 normalise the point 
        pt_normalise = hp.normalise(pt,bbox,normalisation)

        #3 create the features vectors for each frame
        if points == True:
            pt_df = hp.create_df_features_points(pt_normalise,conf_type)
            if with_conf==True:
                pt_df.T[pt_df.T["confidence"]<=0.5] = np.nan
                pt_df.T[pt_df.T["confidence"].isna()] = np.nan

        if angles == True:
            ang_df=hp.create_df_features_angles(pt_normalise,type_norm,conf_type)
            if with_conf==True:
                ang_df.T[ang_df.T["confidence"]<=0.5] = np.nan
                ang_df.T[ang_df.T["confidence"].isna()] = np.nan

        if segments == True:
            seg_df = hp.create_df_features_segments(pt_normalise,type_norm,conf_type)
            if with_conf==True:
                seg_df.T[seg_df.T["confidence"]<=0.5]=np.nan
                seg_df.T[seg_df.T["confidence"].isna()] = np.nan
        
        pose = (pt_df,ang_df,seg_df)
        pose_lab = FILE[FILE.find("n/")+2 : FILE.rfind("/")] # find the label in the string

        #4 get one big list with all the features
        poses.append(pose)
        lab_all_files.append(pose_lab)
        
    # initialisation 
    similarity=[]
    
    for index,FILE in enumerate(ALL_FILES):
        # get the result which represent the pose with the most similarity with it 
        # remove the pose that we are working with
        pose_input = poses[index]
        pose_label_input = lab_all_files[index]
        poses.pop(index)
        lab_all_files.pop(index)

        if len(poses)>=0: # Flag to stop the process when all poses have been processed

            # initialise loss

            loss_pts = []
            loss_ang = []
            loss_seg = []
            true_label = []
            voted_lab = []
            loss_p_norm = []
            loss_a_norm = []
            loss_s_norm = []
            
            # comparison with all the other pose
            for p,dst_lab in zip(poses,lab_all_files):
                #points loss
                if (pose_input[0].empty or p[0].empty) is False:
                    loss_p = hp.calculate_loss_with_confidence(pose_input[0],p[0],loss_type,loss_conf_type)
                    loss_pts.append(loss_p)
                #angles loss
                if (pose_input[1].empty or p[1].empty) is False:
                    loss_a = hp.calculate_loss_with_confidence(pose_input[1],p[1],loss_type,loss_conf_type)
                    loss_ang.append(loss_a)
                #segments loss
                if (pose_input[2].empty or p[2].empty) is False:
                    loss_s = hp.calculate_loss_with_confidence(pose_input[2],p[2],loss_type,loss_conf_type)
                    loss_seg.append(loss_s)
                
                # add both the real label and the vote
                true_label.append(pose_label_input)
                voted_lab.append(dst_lab)

            # MinMax normalisation
            if points == True:
                loss_p_norm = (loss_pts-np.min(loss_pts))/(np.max(loss_pts)-np.min(loss_pts))
            if angles == True:
                loss_a_norm = (loss_ang-np.min(loss_ang))/(np.max(loss_ang)-np.min(loss_ang))
            if segments == True:
                loss_s_norm = (loss_seg-np.min(loss_seg))/(np.max(loss_seg)-np.min(loss_seg))
                
            # compute all the loss put it in a table with all the other label and then take top five and see the neighbour
            loss = pd.DataFrame([loss_p_norm,loss_a_norm,loss_s_norm]).T.mean(axis=1) # regrouping loss
            result = pd.DataFrame([true_label,voted_lab, loss],index=["label_true","label_pred","loss"]).T 
            best_lab = result.sort_values("loss").head(k).groupby("label_pred").count().sort_values("loss",ascending=False).index[0] # sorting the vote and take the best score
            similarity.append(best_lab)

            # insert back the tested file and pose
            poses.insert(index,pose_input)
            lab_all_files.insert(index,pose_label_input)

    # recall calculus
    recall = recall_score(lab_all_files,similarity,labels=LABEL_LIST,average="micro")



    if visu==True:
            # End timer
        end = time.time()
        print("The time to execute the Knn modified method is : ")
        print(str(end - start)+ " seconds")
        print("Recall is equal to: "+str(recall))
        #plot the confusion matrix
        cm=confusion_matrix(lab_all_files,similarity,labels=LABEL_LIST,normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LABEL_LIST)
        fig, ax = plt.subplots(figsize=(5,5))
        disp.plot(ax=ax)
        plt.xticks(rotation=30,ha='right')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig('Figures/knn_modif_'+str(k)+'norma_'+str(normalisation)+'_conf_'+conf_type+'.png', pad_inches=5)
        plt.show()

    return recall
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def ann(LABEL_LIST,ALL_FILES,k,normalisation,type_norm,points,angles,segments,conf_type,loss_ann,n_tree,norm_nan,with_conf,visu=True):

    ''' 
    INPUT: 
    This function takes as argument a list of the unique label (LABEL_LIST) [string], the list with all the files that we want to compare (ALL_FILES)[string],
    the number of neighbors voting to choose the label (k)[int], the type of normalisation used on the skeletton keypoints (normalisation) [string], 
    the type of norm used to calculate angles and segments (type_norm)[int],flags if we take in consideration points,angles or segments [bool],
    the choice of the method to treat the confidence (conf_type)[string],the choice of the method to treat the nan (norm_nan)[string], 
    number of tree (n_tree)[int],the type of loss used to calculate the distance btw neighbors (loss_ann)[string],
    flags if we take in consideration the conf [bool],flags if we take in consideration a visualisation of the matrix of confudion [bool]
    Output: 
    Recall matrix of the comparison between all poses and the confusion matrix for the different classes
    '''
    # Start timer
    start = time.time()

    #initialisation
    labels = []
    conf_poses = []
    poses = []
    for FILE in ALL_FILES:
            #1 get the point and the Json from the file 
            pt,bbox = hp.get_pose_fromJson_JPG(FILE)
            #2 normalise the point 
            pt_normalise = hp.normalise(pt,bbox,normalisation)
            #3 create the features vectors for each frame
            pose = hp.create_df_features(pt_normalise,conf_type,type_norm,points,angles,segments)
            pose.T[pose.T["confidence"].isna()] = np.nan  # check that the data are consistent 
            
            conf_pose = pose.T.confidence.mean() # calculate the mean for each pose
            conf_poses.append(conf_pose)

            pose = np.array(pose.T.metric)
            pose_label = FILE[FILE.find("n/")+2 : FILE.rfind("/")] # find the label in the string
            #4 get one big list with all the features
            poses.append(pose)
            labels.append(pose_label)

    poses_df = pd.DataFrame(poses) # creates a dataset from the points
    # if more than the half of the point are not detected I don't take the pose
    mask = np.sum(poses_df.isnull(),axis=1)<=poses_df.shape[1]//2
    ALL_FILES_masked = [FILE for FILE,bool in zip(ALL_FILES,mask) if bool]
    poses_df = poses_df.dropna(thresh=poses_df.shape[1]//2)

    if norm_nan == "mean_on_row":
        # take the mean for each feature dimension and replace the nan with it
        mean_by_dim = poses_df.mean(axis=0)
        poses = np.array(poses_df.fillna(mean_by_dim))

    elif norm_nan == "fill_zero":
        # replace the nan with only 0
        poses = np.array(poses_df.fillna(0))
    
    elif norm_nan == "median":
        # take the median for each feature dimension and replace the nan with it
        median_by_dim = (poses_df.max(axis=0)-poses_df.min(axis=0))/2
        poses = np.array(poses_df.fillna(median_by_dim))

    elif norm_nan == "max":
        # take the max for each feature dimension and replace the nan with it
        max_by_dim = poses_df.max(axis=0)
        poses = np.array(poses_df.fillna(max_by_dim))
    
    elif norm_nan == "min":
        # take the min for each feature dimension and replace the nan with it
        min_by_dim = poses_df.min(axis=0)
        poses = np.array(poses_df.fillna(min_by_dim))

    elif norm_nan == "random_uni":
        # replace the nan with a random values from a uniform distribution
        max_by_dim = poses_df.max(axis=0)
        min_by_dim = poses_df.min(axis=0)
        poses = np.array(poses_df.fillna(random.uniform(min_by_dim, max_by_dim)))
    
    elif norm_nan == "random_gauss":
        # replace the nan with a random values from a gaussian distribution
        mean_by_dim = poses_df.mean(axis=0)
        std_by_dim = poses_df.std(axis=0)
        poses = np.array(poses_df.fillna(random.gauss(mean_by_dim, std_by_dim)))

    
    f = poses[0].shape[0]  # Length of item vector that will be indexed

    # use Annoy Library 
    # https://github.com/spotify/annoy accessed on 2022

    t = AnnoyIndex(f,loss_ann) #create the tree

    for idx,p in enumerate(poses): # add all the items 
        t.add_item(idx, p)
    t.build(n_tree) #build the tree
    labels = np.array(labels) # to array in order to query the neighbors labels

    #initialisation 
    true_label = []
    pred_label = []

    for idx,pose in enumerate(ALL_FILES_masked):
        #k+1 as the first neighbor is the point itself
        neighbors,distance = t.get_nns_by_item(idx,20,search_k=-1,include_distances=True) # query the distances and the nearest neighbors
        neigh_conf_mean = np.array(conf_poses)[neighbors] # get the confidence
        if with_conf == True:
            df_ann = pd.DataFrame({"neigh":neighbors,"distance":distance,"conf":neigh_conf_mean})
            df_ann["distance_with_conf"] = df_ann.distance/df_ann.conf # distances weighted based on the confidence 
            neighbors = df_ann.sort_values("distance_with_conf")["neigh"].values # sorting 
        neigbors_labels = labels[neighbors[1:k+1]] #extract the k nearest neighbors
        neigh_count = np.unique(neigbors_labels,return_counts=True) # count the voting
        most_neigh_ind = neigh_count[1].argmax() 
        voted_label = neigh_count[0][most_neigh_ind] # define the label with the highest score
        true_label.append(labels[idx])
        pred_label.append(voted_label)
        
        
    recall = recall_score(true_label,pred_label,labels=LABEL_LIST,average="micro")
    if visu == True:
        print("Recall is equal to: "+str(recall))
        # End timer
        end = time.time()
        print("The time to execute the Ann method is : ")
        print(str(end - start)+ " seconds")
            
        #plot confusion matrix
        cm = confusion_matrix(true_label,pred_label,labels=LABEL_LIST,normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LABEL_LIST)
        fig, ax = plt.subplots(figsize=(5,5))
        disp.plot(ax=ax)
        plt.xticks(rotation=30,ha='right')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig('Figures/Ann_'+str(k)+'norma_'+str(normalisation)+'_conf_'+str(conf_type)+"_"+str(with_conf) +'.png', pad_inches=5)
        print(with_conf)
        plt.show()
    return recall
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def pas_eval(LABEL_LIST,ALL_FILES,k,thresholdAngle,sub_method,normalisation,conf_type,type_norm,with_conf):

    ''' 
    INFO: 
    This methods is based on the paper "Dance self-learning application and its dance pose evaluations”. So, the numenclature have been kept. "T" stands for the teacher related 
    information ( in our case it corrresponds to the data that we are currently querying), inversely "S" stands ffor students related informations.

    INPUT: 
    This function takes as argument a list of the unique label (LABEL_LIST) [string], the list with all the files that we want to compare (ALL_FILES)[string],
    the number of neighbors voting to choose the label (k)[int], the type of normalisation used on the skeletton keypoints (normalisation) [string], 
    the type of norm used to calculate angles and segments (type_norm)[int],the choice of the method to treat the confidence (conf_type)[string],
    the choice of size of the threshold angle (thresholdAngle)[int], the type of submethod used  (sub_method)[string]

    Output: 
    Recall matrix of the comparison between all poses and the confusion matrix for the different classes
    '''

    # Start timer
    start = time.time()

    labels=[FILE[FILE.find("n/")+2 : FILE.rfind("/")] for FILE in ALL_FILES] # get the labels

    similarity=[]
    # create features on all the files
    for index,FILE_T in enumerate(ALL_FILES):
        label_T=FILE_T[FILE_T.find("n/")+2 : FILE_T.rfind("/")] # label of the teacher 
        ALL_FILES.pop(index) 

        # initialisation
        pas_list=[]
        true_label=[]
        voted_label=[]
        # comparison with all the other pose
        for FILE_S in ALL_FILES:
            label_S=FILE_S[FILE_S.find("n/")+2 : FILE_S.rfind("/")] # label of the student
            pas_res=pas.get_pas(FILE_S,FILE_T,type_norm,sub_method,thresholdAngle,normalisation,conf_type,with_conf,ratio_pas=0.5,visu=False)
            pas_list.append(pas_res)
            true_label.append(label_T)
            voted_label.append(label_S)
                
        result=pd.DataFrame([true_label,voted_label, pas_list],index=["label_true","label_pred","pas"]).T
        #sorting and get the most fitted label
        label_neigh = result.sort_values("pas",ascending=False).head(k).groupby("label_pred").count().sort_values("pas",ascending=False).index[0]
        similarity.append(label_neigh)
        ALL_FILES.insert(index,FILE_T)
    # calcul the recall
    recall =recall_score(labels,similarity,labels=LABEL_LIST,average="micro")
    print("Recall is equal to: "+str(recall))
    
    # End timer
    print("The time to execute the PAS modified method is : ")
    end = time.time()
    print(str(end - start) + " seconds")

    # plot the confusion matrix
    cm=confusion_matrix(labels,similarity,labels=LABEL_LIST,normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LABEL_LIST)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax)
    plt.xticks(rotation=30,ha='right')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('Figures/pas_'+str(k)+'norma_'+str(normalisation)+'_'+str(conf_type)+'_'+str(sub_method)+'.png',  pad_inches=5)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def query_ann(ALL_FILES,QUERY_PATH,k,normalisation,type_norm,points,angles,segments,conf_type,loss_ann,n_tree,with_conf):


    '''
    This function takes as argument the list with all the files that we want to compare (ALL_FILES)[string],
    the number of neighbors voting to choose the label (k)[int], the type of normalisation used on the skeletton keypoints (normalisation) [string], 
    the type of norm used to calculate angles and segments (type_norm)[int],flags if we take in consideration points,angles or segments [bool],
    the choice of the method to treat the confidence (conf_type)[string],the choice of the method to treat the nan (norm_nan)[string], 
    number of tree (n_tree)[int],the type of loss used to calculate the distance btw neighbors (loss_ann)[string],
    flags if we take in consideration the conf (with_conf)[bool]
    
    OUTPUT:
    return k nearest neighbors of the query path
    '''

    #import the image and extract skeletons with Open PifPaf
    capture = cv2.VideoCapture(QUERY_PATH)
    _,image = capture.read()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # extract squeleton with 
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16',json_data=True)
    predictions, _ , _ = predictor.numpy_image(image)
    poses_df=predictions[0]
    kp=poses_df.get("keypoints")
    keypoints=np.asarray(kp).reshape(17,3)
    bbox_T=np.asarray(poses_df.get("bbox"))
    keypoints=pd.DataFrame(keypoints,columns=["x","y","confidence"])
    keypoints[keypoints.confidence==0]=np.nan
    kpCenterBbox_T = hp.normalise(keypoints,bbox_T,normalisation)
    # made a vector v with the feature
    v=np.array(hp.create_df_features(kpCenterBbox_T,conf_type,type_norm,points,angles,segments).T.metric)

    # treatment of the others FILES

    #initialisation
    poses=[]
    labels=[]
    conf_poses=[]

    for FILE in ALL_FILES:
        #1 get the point and the Json from the file 
        point,bbox=hp.get_pose_fromJson_JPG(FILE)
        #2 normalise the point 
        pt_normalise=hp.normalise(point,bbox,normalisation)
        pose=hp.create_df_features(pt_normalise,conf_type,type_norm,points,angles,segments)
        conf_pose=pose.T.confidence.mean()
        conf_poses.append(conf_pose)
        pose=np.array(pose.T.metric)
        pose_label=FILE[FILE.find("n/")+2 : FILE.rfind("/")] # find the label in the string
        #4 get one big list with all the features
        poses.append(pose)
        labels.append(pose_label)
        
    poses_df = pd.DataFrame(poses)

    # mean on each dimension if nan
    mean_by_dim=poses_df.mean(axis=0)
    poses=np.array(poses_df.fillna(mean_by_dim))

    # if more than the half of the point are not detected I don't take the pose
    mask=np.sum(poses_df.isnull(),axis=1)<=poses_df.shape[1]//2
    ALL_FILES_masked=[FILE for FILE,bool in zip(ALL_FILES,mask) if bool]
    poses_df=poses_df.dropna(thresh=poses_df.shape[1]//2)

    f = poses[0].shape[0]  # Length of item vector that will be indexed
    
    t = AnnoyIndex(f, loss_ann)
    for idx,p in enumerate(poses):
        t.add_item(idx, p)
    t.build(n_tree)

    #deal with the nan of the vector
    vect_df = pd.DataFrame(v)
    # mean on each dimension if nan
    v=np.array(vect_df.fillna(mean_by_dim))

    # get the neighbors of the vector feature of my personal pose 
    neighbors,distance=t.get_nns_by_vector(v,20,search_k=-1,include_distances=True)
    neigh_conf_mean=np.array(conf_poses)[neighbors]
    if with_conf==True:
        df_ann=pd.DataFrame({"neigh":neighbors,"distance":distance,"conf":neigh_conf_mean})
        df_ann["distance_with_conf"]= df_ann.distance/df_ann.conf
        neighbors=df_ann.sort_values("distance_with_conf")["neigh"].values
    neighbors_sort=neighbors[0:k]
    img_T = cv2.imread(QUERY_PATH)
    plt.imshow(img_T)


    # plot the result
    fig, ax = plt.subplots(int(k/2),2,figsize=(6,8),squeeze=False)
    for i,FILE in enumerate(np.array(ALL_FILES_masked)[neighbors_sort]):
        FILE=FILE.replace(".json", ".jpg")
        FILE_IMAGE=FILE.replace("/Json/", "/Image/")
        plt.subplot(int(k/2),2,i+1)
        img = cv2.imread(FILE_IMAGE)
        plt.imshow(img)
    fig.tight_layout()  
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def ann_video(PATH_DATA,LABEL_LIST_VIDEO,ALL_FILES_VIDEO,clust_flag,clust_type,with_conf,nb_min_points_detected,labels_video):

    '''
    Input: the list of labels (LABEL_LIST) [list of string], the minimum number of points that needs to be detected 
    to consider the frame (nb_min_points_detected) [int], flag for the clustering (clust_flag) [bool],
    type of the clustering (either kmeans, minMax_xy or time_segmentation) [string]

    Output: general recall for the chosen labels and matrix of confusion
    '''

    #initialisation

    all_features_group=pd.DataFrame()
    all_conf_group=pd.DataFrame()
    

    for lab in LABEL_LIST_VIDEO:
        print(lab)
        BASE_PATH = PATH_DATA+"/Videos/"+lab
        # get all the files for the label
        FILES=[f for f in listdir(BASE_PATH) if f.endswith(".json")]
        # sorted on different way with a clustering ot not 
        features_group_by_lab=pd.DataFrame([hp.video_dimensional_analysis(BASE_PATH+"/"+f,nb_min_points_detected,clust_flag,clust_type,visu=False) for num,f in enumerate(FILES)])
        # stores for the features and confidences in a Dataframes of Dataframes
        features_group_by_lab=features_group_by_lab.rename({0:"features",1:"confidences"},axis=1)
        # construct one unique dataset for all the features and confidence 
        all_features_group=pd.concat([all_features_group,pd.DataFrame(np.vstack(features_group_by_lab["features"]))])
        all_conf_group=pd.concat([all_conf_group,pd.DataFrame(np.vstack(features_group_by_lab["confidences"]))])


    # uses Ann function  with the best critereon find with the images 
    norm_nan = "mean_on_row"
    loss_ann="euclidean"
    n_tree=2
    k=5

    # if more than the half of the point are not detected I don't take the video
    mask=np.sum(all_features_group.isnull(),axis=1)<=all_features_group.shape[1]//2
    ALL_FILES_masked=[FILE for FILE,bool in zip(ALL_FILES_VIDEO,mask) if bool]
    all_features_group=all_features_group.dropna(thresh=all_features_group.shape[1]//2)
    # mean on each dimension if nan
    mean_by_dim=all_features_group.mean(axis=0)
    poses=np.array(all_features_group.fillna(mean_by_dim))
    


    f = poses[0].shape[0]  # Length of item vector that will be indexed

    t = AnnoyIndex(f,loss_ann)
    for idx,p in enumerate(poses):
        t.add_item(idx, p)
    t.build(n_tree)

    labels_video=np.array(labels_video)
    true_label=[]
    pred_label=[]

    for idx,pose in enumerate(ALL_FILES_masked):

        #k+1 as the first neighbor is the point itself
        neighbors,distance=t.get_nns_by_item(idx,20,search_k=-1,include_distances=True)
        neigh_conf_mean=np.array(all_conf_group.mean(axis=1).reset_index(drop=True))[neighbors]
        if with_conf==True:
            df_ann=pd.DataFrame({"neigh":neighbors,"distance":distance,"conf":neigh_conf_mean})
            df_ann["distance_with_conf"]= df_ann.distance/df_ann.conf
            neighbors=df_ann.sort_values("distance_with_conf")["neigh"].values
        neigbors_labels=labels_video[neighbors[1:k+1]]
        neigh_count = np.unique(neigbors_labels,return_counts=True)
        most_neigh_ind = neigh_count[1].argmax()
        label_predicted = neigh_count[0][most_neigh_ind]
        true_label.append(labels_video[idx])
        pred_label.append(label_predicted)
        

    recall =recall_score(true_label,pred_label,labels=LABEL_LIST_VIDEO,average="micro")

    print("Recall is equal to: "+str(recall))
        
    #plot confusion matrix
    cm=confusion_matrix(true_label,pred_label,labels=LABEL_LIST_VIDEO,normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LABEL_LIST_VIDEO)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax)
    plt.xticks(rotation=30,ha='right')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('Figures/Video_Ann_'+str(clust_type)+'.png', pad_inches=5)
    plt.show()
    
    


