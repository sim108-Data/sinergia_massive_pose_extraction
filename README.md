# Sinergia,massive pose extraction

This repository contains the following main folders and files :

* **Data** : regroups all the Data used in this project (FrameToFrame: regroups the files of our self-made dataset both Json and JPG, Videos regroups the same files as the previous datasets but keeps contains the full video and the JSON output of Open PifPaf )

* **Figures** : regroups all outputs of the methods used in the code for each tasks ( raw / normalised / confidence ) and auxilliary visualization output. (**Ann**/ **Knn**/**Pas** / **Videos**)

* **others** : contains all python files used to compute function necessary at our **methods.py** ( **helpers.py** : all function helping the implementation, **pas.py** : all function used to implement the Position and angle similarity based on the [paper](https://dl.acm.org/doi/10.1145/3410530.3414402) presented in the report and **visualisation.py** : function in order to plot images as examples )

* **methods.py** : implements all methods used in the project and also a function used to tune our hyperparameters. 

* **main.py** : define the argument parsers and call the methods implemented  in "**methods.py**"

## Data
The data used in this project comes from the [DeepMind 700 Kinetics human action dataset](https://arxiv.org/abs/2010.10864). Updated in 2020, this dataset contains at least 700 video clips extracted from Youtube videos for each of the classes. To define how well those methods quantify the difference between poses and with the help of Kinetics 700 labels, a self-made dataset is created. To map the data from Kinetics and outputs from Openpifpad the **train.csv** is used.

## Getting Started

First, you will need to have " git " install on your computer to be able to clone the repository on your computer. You will also need to have python installed. If this is not the case you can download it from here: [Python]( https://www.python.org/downloads/ )

## Dependencies and Installing

The code is using some external libraries: "Numpy", "Pandas" to analyze data, "Matplotlib" and "Seaborn" to visualize the output of the analysis, "[opencv](https://opencv.org/)" to deal with image and videos, [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)" to apply dimensional reduction, "[tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html)" for clustering with Kmean,  and "[openpifpaf](https://openpifpaf.github.io/intro.html)" to extract the skeletons. 

 **requirements.txt** file:

```
pip install -r requirements.txt
```


### Executing program

To execute the program you need to follow the steps described below:
* Clone the repository on your computer
* Get the Python libraries needed
* Enter in the repository with this command (Windows) or equivalent code line for Mac and Linux
```
cd  sinergia_massive_pose_extraction
```
* Run Knn best result as follow with the command line:
```
python main.py  --method_name knn_modified  --conf_type none --normalisation bbox_ratio_kept --points True --angles True
```
* Run Ann's best result as following with the command line:
```
python main.py  --method_name ann --points True --angle True --normalisation bbox_ratio_kept_center_core --norm_nan mean_on_row --with_conf True --conf_type mean --loss_conf_type mean --type_norm 2 --k 5
```
* Run PAS best result as following with the command line:
```
python main.py  --method_name pas_eval  --conf_type mean --normalisation bbox_ratio_kept  --thresholdAngle 15 --sub_method core --with_conf True

```
* Query ann as following with the command line for a my image of basketaball dribbling :
```
python main.py  --method_name query_ann --points True --angles True --normalisation bbox_ratio_kept_center_core --conf_type mean --loss_conf_type mean --norm_nan mean_on_row --k 6 --with_conf True

```

* Query ann as following with the command line for a my image of tennis :
* 
```
python main.py  --method_name query_ann --points True --angles True --normalisation bbox_ratio_kept_center_core --conf_type mean --loss_conf_type mean --norm_nan mean_on_row --k 6 --with_conf True --query_path "my_Data\my_tennis.jpg"

```

* Query ann as following with the command line for a test image of tennis :
```
python main.py  --method_name query_ann --points True --angles True --normalisation bbox_ratio_kept_center_core --conf_type mean --loss_conf_type mean --norm_nan mean_on_row --k 6 --with_conf True --query_path "my_Data\test_tennis.jpg"

```

* Query ann on videos as following with the command line :
```
python main.py  --method_name ann_video --clut_type minMax_xy

```



## Authors

Simon Dayer

## License
Copyright © 2021 Simon Dayer. All rights reserved.
