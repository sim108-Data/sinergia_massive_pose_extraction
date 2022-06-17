#librairy 
import argparse 
from os import listdir

# other files
from methods import *

def main():

    parser = argparse.ArgumentParser(description='Query images metrics testing')
    #data
    parser.add_argument('--data_path', type=str, default='Data/',
                        help = 'Path of data.')
    parser.add_argument('--query_path', type=str, default='my_Data\Frame_baskdribble.jpg',
                        help = 'Path of the picture we want to query.')

    #choice of the method
    parser.add_argument('--method_name', type=str, default='ann',
                        help='input the method (default: KNN)')

    #choice of the arguments
    parser.add_argument('--points', type=bool, default=False,
                        help='take keypoint positions in count')
    parser.add_argument('--angles', type=bool, default=False,
                        help='take angles in count')
    parser.add_argument('--segments', type=bool, default=False,
                        help='take segments size in count')
    parser.add_argument('--normalisation', type=str, default="bbox_ratio_kept",
                        help='choose how the skeleton from the image are normalise')
    parser.add_argument('--conf_type', type=str, default="mean",
                        help='choose how the confidence is treated')
    parser.add_argument('--loss_conf_type', type=str, default="mean",
                        help='choose how the confidence is treated when computing the loss')
    parser.add_argument('--type_norm', type=int, default=2,
                        help='choose the norm used to calculate angle and segments [default: L2]')
    parser.add_argument('--k', type=int, default=5,
                        help='k number of neigboour to take in order to compute Top-k recall')

    #for ann additonal arguments
    parser.add_argument('--norm_nan', type=str, default="mean_on_row",
                        help='nan replacement ')
    parser.add_argument('--with_conf', type=bool, default=False,
                        help='take or not the conf ')
    parser.add_argument('--n_tree', type=int, default=2,
                        help='number of tree for knn')
    parser.add_argument('--loss_ann', type=str, default="euclidean",
                        help='metric for knn method')

    #for knn_modified additonal arguments
    parser.add_argument('--loss_type', type=str, default="euclidean_distance_mean",
                        help='how to calculate the loss')

    #for PAS additonal arguments
    parser.add_argument('--thresholdAngle', type=int, default=30,
                        help='thresholdfor the angle in JASS')
    parser.add_argument('--sub_method', type=str, default="align_torso",
                        help='disposition of the ')
    # for Ann videos additional arguments
    parser.add_argument('--clust_type', type=str, default="time_segmentation",
                        help= 'choose how to divide the video into 4 frames ')
    parser.add_argument('--clust_flag', type=bool, default=False,
                        help='take or not a clust ( mandatory for Kmeans method) ')
    parser.add_argument('--nb_min_points_detected', type=int, default=10,
                        help='threshold on the minimum number of points that need to be detected in order to take the pose into consideration')
    

    parser.add_argument('--type_compar', type=str, default="none",
                        help= 'extract the recall for all the combinaison of points/angles/segments for different settings [none/norm/norm_withconf]')
    args = parser.parse_args()
    
    # list with all the files stored  
    
    #frame to frame
    LABEL_LIST=listdir(args.data_path+"/FrameToFrame/Json/")  
    ALL_FILES=[]
    for i in LABEL_LIST:
        FILES=listdir((args.data_path+"/FrameToFrame/Json/"+i))  
        FILES_=[args.data_path+"/FrameToFrame/Json/"+i+"/"+s for s in FILES if s[-1]=="n"]
        ALL_FILES+=FILES_

    #video

    LABEL_LIST_VIDEO=listdir(args.data_path+"/Videos")
    ALL_FILES_VIDEO=[]
    labels_video=[]
    for lab in LABEL_LIST_VIDEO:
        BASE_PATH=args.data_path+"/Videos/"+lab
        FILES=[f for f in listdir(BASE_PATH) if f.endswith(".json")]
        ALL_FILES_VIDEO+=FILES
        labels=[lab]*len(FILES)
        labels_video+=labels




        
    # Define the method 
    if args.method_name == 'knn_modified':
        knn_modified(LABEL_LIST,ALL_FILES,args.loss_type,args.k,args.normalisation,args.type_norm,args.points,args.angles,args.segments,args.conf_type,args.with_conf,args.loss_conf_type)
    elif args.method_name == 'ann':
        ann(LABEL_LIST,ALL_FILES,args.k,args.normalisation,args.type_norm,args.points,args.angles,args.segments,args.conf_type,args.loss_ann,args.n_tree,args.norm_nan,args.with_conf)
    elif args.method_name == 'pas_eval':
        pas_eval(LABEL_LIST,ALL_FILES,args.k,args.thresholdAngle,args.sub_method,args.normalisation,args.conf_type,args.type_norm,args.with_conf)
    elif args.method_name == 'query_ann':
        query_ann(ALL_FILES,args.query_path,args.k,args.normalisation,args.type_norm,args.points,args.angles,args.segments,args.conf_type,args.loss_ann,args.n_tree,args.with_conf)
    elif args.method_name ==  'ann_video':
        ann_video(args.data_path,LABEL_LIST,ALL_FILES_VIDEO,args.clust_flag,args.clust_type,args.with_conf,args.nb_min_points_detected,labels_video)
    elif args.method_name == 'comparison_method_ann':
        comparison_recall(LABEL_LIST,ALL_FILES,args.type_compar,"ann")
    elif args.method_name == 'comparison_method_knn':
        comparison_recall(LABEL_LIST,ALL_FILES,args.type_compar,"knn_modified")
    elif args.method_name == 'comparison_method_pas':
        comparison_recall(LABEL_LIST,ALL_FILES,args.type_compar,"pas")
    else:
        raise Exception("The chosen method is not implemented.")


    
if __name__ == '__main__':
    main()
    
