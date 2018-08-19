# Video Anomaly Detection

This is code for BMVC 2018 submission #237:
*Classifier Two-Sample Test for Video Anomaly Detections*

####Requirements
The code is written in Matlab, and used with Matlab 2017a in a laptop with MacOS system

#### Run the script
The experiment and parameters are included in Run_script.m. This will generate a series of score files and features. Make sure to put the Avenue dataset in prepared folders before using, and see the txt file in PrepareData folder to put in imagenet.mat for feature extraction. 

The script evaluation.m will read in the generated score files and compare with ground truth provided in Avenue dataset, to compute and display the AUC score. Individual AUC scores are also avaliable but not displayed. 
