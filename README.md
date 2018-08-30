# Video Anomaly Detection

This repo contains the example code for the paper *Classifier Two-Sample Test for Video Anomaly Detections*(BMVC 2018)

(Links and Citation coming soon)

## Using the code
* Download package:
```bash
$ git clone https://github.com/MYusha/Video-Anomaly-Detection
```
* Assume the default path is `Video-Anomaly-Detection/pipeline`.

**Requirements:** The code is written in Matlab 2017a, and used with laptop with MacOS. Please first install
[liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
matlab. And put [imagenet](http://www.vlfeat.org/matconvnet/pretrained/) inside `PrepareData/Appearance_feature/` for appearance feature extraction.

**Specification:** The functions for motion features computation are modified from re-implementation in https://github.com/gongruya/abnormality-detection.

## Dataset preparation
Please put the
[Avenue datatset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
from CUHK inside `Avenue_Dataset` folder.

## Instuctions
**Generate scores:** 
The experiment and parameters are included in `pipeline/Run_script.m`. Running this script will generate a series of features and anomaly score files for the videos.
  
**Compute AUC:** 
The script evaluation.m will read in the generated score files and compare with ground truth provided, to compute and display the AUC score. Individual AUC scores are also avaliable but not displayed. 
