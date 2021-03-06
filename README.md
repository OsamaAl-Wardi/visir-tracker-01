# Practical Assignment 1
Please put your name (or names if you work in a group) here:   
**Name**: Osama Al-Wardi and Melvin Wolf
## Python3 opencv setup
You need a standard python3 setup. You obviously need python3 and pip3 installed.

After that you need numpy:
```
sudo pip3 install numpy
```
and opencv
```
pip3 install opencv-python
```
## Running the code
Just run this command:
```
python3 face_detection.py
```
## Results:
After running the code you can see the normal camera stream and the fps on the terminal output. Then press q and you will have the face detection algorithm running and the fps on the terminal output. If you have any questions write me on o.alwardi@jacobs-university.de **Please Note** this code is written and tested on an Ubuntu 18.04 operating system. If you're running windows the code might not work due the paths using '\\' instead of '/'. Let me know if you get problems with that.
## Problem 1.1
### Calculate Frames-per-Second (FPS) (Points 30)
1. Fork the current repository
2. Study the new framework-code of 
    - main.cpp
3. Check that the code is running correctly: it should show the video stream from the web-camera of your laptop.
4. Calculate average fps and print it to console every 2 seconds. Compare Debug and Release versions.
### Note
MacOS users may need to launch the application with administrator right, to grant access to the web-camera.

## Problem 1.2
### Face detection (Points 70)
1. Read the OpenCV documentation about Viola-Jones face detector: [Cascade Classifier](https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html)  
2. Implement face detection for the video strem from the web-camera using the ```cv::CascadeClassifier``` class.
3. Measure the FPS one more time. How FPS changed after incorporating the face detection into the framework?
### Note
Please do not copy-paste the example code from the OpenCV documentation, but try to understand the example code and implement the solution to the problem by yourself.
