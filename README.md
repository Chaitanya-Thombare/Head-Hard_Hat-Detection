# Hard_Hat/Head Detection
---
## Introduction
This repository is to detect whether the person within the camera vision has worn hard_hat(construction site helmet) or not. For that, a custom yolov3 tiny is trained. The dataset provided for training has 5000 labelled images, and model was trained for 6000 epochs.

## Model Training
The files required for yolov3 tiny model(416) are:
* yolov3 tiny [cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3-tiny.cfg)
* yolov3 tiny [custom cfg]()
* yolov3 pretrained [convolution weights](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)

Final output weights files are [here](https://drive.google.com/drive/folders/1qjQC5L_eWiJJ9GhjSnAZXc4dVR3eQzmY?usp=sharing)

## Model Performance
Input Images:

![alt text](https://github.com/Chaitanya-Thombare/hard_hat-head-detection/blob/main/media/inp1.png)

![alt text](https://github.com/Chaitanya-Thombare/hard_hat-head-detection/blob/main/media/inp2.png)

Outputs Images:

![alt text](https://github.com/Chaitanya-Thombare/hard_hat-head-detection/blob/main/media/out1.png)

For this image, the color of bounding box is determined by the confidence of predictions given by YOLO model.

![alt text](https://github.com/Chaitanya-Thombare/hard_hat-head-detection/blob/main/media/out2.png)

For this image, the color of bounding box is near to the color of hard_hat. For that, K-Means clustering algorithm is used.
For K-Means model, each pixel is an object clustering is done on bases of RGB values of each pixel. Image passed to the K-Means models is cropped detection again cropped 10% from top left and right and 30% from bottom to maximize the area covered by the hard hat in the image.
