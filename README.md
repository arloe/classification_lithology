### Lithology Classification
Predict Lithology using resnet-type classification model

### Introduction
I worked on a project related to oil exploration for company SK Innovation (2020.02 - 2020.12).
The lithological information is the most important factor in oil exploration from geo-science point of view. Therefore, predicting the lithology is a very important matter.

For securty reasons, the code was uploaded using [Xeek](https://xeek.ai/challenges/force-well-logs/data) data that is public, not the data of SK Innovation.

The physical properties (eg. NPHI, RHOB, ...) wer used as input features. The data is called well-log and is in las format. In this code, it is assumed that the data is in csv format.

### Usage
```bash
$ python train.py -c config.json -d 0 -i first_train
```
Argument `-c` means the name of configuration file. Also, `-d` is the index of the GPU, and `-i` is the name of the folder where the learned model and log files are put.

