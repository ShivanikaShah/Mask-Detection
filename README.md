# Mask-Surveillance

A deeplearning based solution for identying individuals not wearing a mask. The solution comprises of three modules:

### 1. Face detection:
An object detection based model to detect faces from images.

### 2. Mask Detection:
A neural network based classification model that can classify masked and unmasked face images

### 3. Face Recognition:
A siamese network that compares the unmasked individuals with the database of individuals and helps recognize the unmasked individuals.


## DEMO
python detect.py --source 0 --weights ../submission/best.pt --conf-thres 0.6 --iou-thres 0.45 --img-size 640
