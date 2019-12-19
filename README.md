# Measure the shakiness of camera

Simple code measures the shakiness\movements of the camera. The code is modified from here<sup>[1](#myfootnote1)</sup>.   Given a video taken by a camera, we want to measure the shakiness of the camera, i.e., the movements of the camera.

The main idea is as follows:

1. Detect key points in *t-1*-th frame and *t*-th frame. 
2. Match the  key points and weed out the bad matches.
3. Compute the homography transformation matrix.

Compute the transformation matrix for every frames and use it to measure the camera movements.  

## Demo

![Demo](.\output\demo.png)



Key points transformation of two videos in every 2 frames.  `dx` refers to  the movements in horizontal direction; `dy` refers to the movements in vertical direction. 

---




<a name="myfootnote1">1</a> .  https://stackoverflow.com/questions/57521164/how-to-detect-if-camera-is-moving-or-shaking-using-homography-opencv-python

