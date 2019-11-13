# Grasping Detection System

### Introduction
This papers investigates several approaches for the detection of graspable areas of an object. In particular, two Resnet-50 and two ResNet-18 models are trained with different configurations: 

1. Pretrained ResNet-18 with a linear activation function
2. Pretrained ResNet-18 with a sigmoid activation function and re-scaling of the labels and output vales
3. Pretrained ResNet-50 with a sigmoid activation function and re-scaling of the labels and output vales
4. Non-pretrained ResNet-50 with a sigmoid activation function and re-scaling of the labels and output vales. 

### Dataset
The models are trained on the Cornell Grasping Dataset, which can be downloaded from http://pr.cs.cornell.edu/grasping/rect_data/data.php

### Results
The result show that all model achieve an accuracy of around 90\%. Out of the four configurations, the ones re-scaling the labels and the output and using a sigmoid activation function (models 2, 3 and 4) resulted rectangles that are better suited for grasping, in terms of orientation, compared to model 1, which did not use such approach.
