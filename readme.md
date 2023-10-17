# Introduction
This code can detect all faces in a video or in a list of images, and enhunce the quality of the faces.

# User guide
The first step after downloading th code, is to download the pretrained super-resolution models from https://drive.google.com/drive/u/1/folders/1T-GfwYdCZv44wENOwEDBVXIjyPHgY-a0 
Put the super-resolution models under PAN/experiments/pretrained_models.

In order to detect and enhance faces in a list of images or a video, the only thing you need to do is to use the function "video_face_detection_and_super_resolution" in detection_and_super_resolution.py.
The input to the function is a list of PIL images. The output of the funtion is a list of PIL images, where each image is an enhunced face.
Possible upscale values for the faces are 2, 3, 4.
A running example is available in main.py.

For more information please read about the possible options in argparser.py
