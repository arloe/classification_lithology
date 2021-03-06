### Lithology Classification
Predict Lithology using resnet-type classification model

### Introduction
I worked on a project related to oil exploration for company SK Innovation (2020.02 - 2020.12).
The lithological information is the most important factor in oil exploration from geo-science point of view. Therefore, predicting the lithology is a very important matter.

For securty reasons, the code was uploaded using [Xeek](https://xeek.ai/challenges/force-well-logs/data) data that is public, not the data of SK Innovation.

The physical properties (eg. NPHI, RHOB, ...) wer used as input features. The data is called well-log and is in las format. In this code, it is assumed that the data is in csv format.

### Usage
1. train
```bash
$ python train.py -c config.json -d 0 -i first_train
```
Argument `-c` means the name of configuration file. Also, `-d` is the index of the GPU, and `-i` is the name of the folder where the learned model and log files are put.

2. test
```bash
$ python test.py -r .\saved\models\0404_211908_cv\model_best.pth -d 0 -t .\data\test_15_9-13.csv -l .\lithology.csv -i 15_9_13
```
The model is selected through `-r`, and `-t` means the directory of the data to be tested. Also, `-l` is basically a csv file for the lithology label.

### Evaluation
The predictive power was evaluated quantitatively and qualitatively. Quantitative evaluation was performed through MAPE and RMSE, and qualitative evaluation was performed through graphs.

 + Quantitative evaluation is below.

![eval1](https://github.com/arloe/classification_lithology/blob/main/img/qualitative_eval.PNG)

 + Qualitative evaluation is below.
 
![eval2](https://github.com/arloe/classification_lithology/blob/main/img/quantitative_eval.PNG) 