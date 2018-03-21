# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* Please refer to this URL for the video. https://youtu.be/DnKthxedemk

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
This model.h5 works with both track1 and track2.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with the following architecture.

0. Preprocessing
- Cropping
- BatchNorm (for normalization)
1. Convolution
  - Conv2D 4 filters with kernel size 5 with stride of (1,2) and valid padding
  - BatchNorm
  - Relu
  - Dropout
2. Convolution
  - Conv2D 4 filters with kernel size 5 with stride of (1,2) and valid padding
  - BatchNorm
  - Relu
  - Dropout
3. Convolution
  - Conv2D 4 filters with kernel size 5 with stride of (1,1) and valid padding
  - BatchNorm
  - Relu
  - Dropout
4. Convolution
  - Conv2D 6 filters with kernel size 3 with stride of (1,1) and valid padding
  - BatchNorm
  - Relu
  - Dropout
5. Convolution
  - Conv2D 8 filters with kernel size 3 with stride of (1,1) and valid padding
  - BatchNorm
  - Relu
  - Dropout
6. Flatten
7. Dense
  - Dense layer with 100 output
  - BatchNorm
  - Relu
  - Dropout
8. Dense
  - Dense layer with 50 output
  - BatchNorm
  - Relu
  - Dropout
9. Dense
  - Dense layer with 10 output
  - BatchNorm
  - Relu
  - Dropout
10. Dense
  - Dense layer with 1 output


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. During training, 20% of the data was held out for validation during training. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was 0.001

#### 4. Appropriate training data

The model was trained and validated on track 1 and track 2 data sets to ensure that the model was not overfitting. I augmented the image data set by mirroring every image, including the left and right camera images. The required angles are taken as the negation of the original angle.

For track 1, I drove the car in the opposite direction to collect data for steering right.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture are as follows,

My first step was to use a convolution neural network model similar to the Nvidia end-to-end self driving paper. I thought this model might be appropriate because the architecture appeared in a published paper that Nvidia claims they are using for their self-driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Then look at the error on the validation set while training on the training set.

#### 2. Final Model Architecture

Refer to earlier description. The Nvidia architecture worked right from the start. The problem I had was that the resulting model.h5 file was very big. So I tried to reduce the architecture size by reducing the number of filters and using a larger stride size for the width. A large stride size reduces the resolution of the image very quickly and allow us to have lesser number of parameters when we reach the densely connected layers.


