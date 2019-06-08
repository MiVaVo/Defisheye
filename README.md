# Image undistortion based on classical straight lines detection

### Main information

This repository contains an algorithm, that is able to some extent identify
the parametrs of radial distortion based on 
straight horizontal lines

### Algorithm explonation
The alrogithm for distortion removel consist of the following steps:
For each configuration of undisortion parametrs:
 - Undistort image 
 - Based on Hough Lines find number of straight lines with predefined 
 angle and predefined length
 - Choose those parametrs of undistorion, based on which the highest number of straight lines is achieved
### Current results
Initial image![Initial img](imgs/not_fisheye_3.jpg?raw=true "Initial image without distortion")

----------

Artificially distorted(fisheyed) image with detected straight lines
![Alt text](result_imgs/Initial_image_with_found_lines.jpg?raw=true "Title")

----------

Undistorted image with found straight lines:
![Alt text](result_imgs/Undistorted_image_with_found_lines.jpg?raw=true "Title")


### Further imporvement
As far as this algorithm is based on classical CV algorithm, namely on Hough Lines,
the process of undistortion rquires callibration of required line width and height for 
different images.


### RUN
```
python main.py
```
