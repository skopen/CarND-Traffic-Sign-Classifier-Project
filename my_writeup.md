# **Traffic Sign Recognition** 

## Writeup

### This is the writeup of my Traffic Sign Classifier project - skopen.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/data_visual.png "Visualization"
[image2]: ./writeup/gray_image.png "Grayscaling"
[image3]: ./writeup/gray_pixel_shifted_image.png "Random Noise"
[image4]: ./web/web_1_speed60.jpeg "Traffic Sign 1"
[image5]: ./web/web_2_side_arrow.jpeg "Traffic Sign 2"
[image6]: ./web/web_3_speed30.jpeg "Traffic Sign 3"
[image7]: ./web/web_4_turn_left.jpeg "Traffic Sign 4"
[image8]: ./web/web_5_caution.jpeg "Traffic Sign 5"
[image9]: ./writeup/orig_image.png "Original image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/skopen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images. I first tried with the given colored (3-channel) images
and was not able to improving accuracy beyond 89%. Intuitively, it seems like most important
aspects of a sign should be readable or learnt from a grayscale image. So I decided to try grayscaling.
And it did improve the accuracy by a few percentage points.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image9]
![alt text][image2]

I also chose to normalize the data by centering it around 128 (0 mean) and scaling it to be between -1 and 1.

I then decided to generate fake data. Fake data was added to intuitively accomodate the fact that
the same sign may be off by a few pixels, but it is still the same sign. I performed a shift of pixels
in the x and y directions. It was performed as following:

- Randomly shift 25% of images right by 1 pixel
- Randomly shift 25% of images left by 1 pixel
- Randomly shift 25% of images up by 1 pixel
- Randomly shift 25% of images down by 1 pixel

Due to randomness of the procedure some images also got shifted diagonally towards the 4 corners, increasing both the diversity
and magnitude of the noise.

![alt text][image3]

The pixel shift needs to be minutely examined to verify.

In effect, it helped improve the validation accuracy. 



The difference between the original data set and the augmented data set is the following:

The augmented data set is twice the size of the original dataset. It contains variants of the same images
but altered in a small way. There are about 8+ variants of the images shifted in different ways
compared to the original images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image                       | 
| Convolution 5x5x6     | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Relu non-linearity							|
| Max pooling	      	| 2x2 stride, output 14x14x6 				    |
| Convolution 5x5x16    | 1x1 stride, output 10x10x16      				|
| RELU		            | Relu non-linearity        					|
| Max pooling			| 2x2 stride, output 5x5x16 (flattened to 400)  |
| Fully connected		| Fully connected, output 120					|
| RELU                  | Relu non-linearity                            |
| Fully connected		| Fully connected, output 84					|
| RELU                  | Relu non-linearity                            |
| Fully connected		| Fully connected, output 43 (logits)			|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I tried various model paramters, but finally settled on the following:

Optimizer: AdamOptimizer

Batch size: 128

Epocs: 12 (tried smaller numbers)

Learning rate: 0.001 (tried smaller number)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.933
* test set accuracy of 1.0 (only the 5 web downloaded images)

Here is my approach:
* I started with a simple approach, which is the use the LeNet architecture as-is with the given 3-channel images.
* I struggled to get 93%+ accuracy on the validation set, even though my training set accuracy was close to 100%
* This made me think if I was overfitting the data
* So I then created a grayscale model and trained on that data. This helped a little bit, but still not good enough.
* Then I tried just pixel shifting to the right as a trial and error. That seemed to have helped.
* So then I tried shifting to the left, up, down and a combination of those.
* I also tried with a lower learning rate, but that did not help much.
* I then tried more epochs, that seems to have helped a bit.
* In the end, I was able to achieve more than 93% validation set accuracy.
* The feature set is highly visual, hence CNN is the right architecture for the problem as evidenced by the results.
* I went ahead with the LeNet architecture since it is a proven architecture and seems to have worked.
* I was especially sure of the architecture after I got 100% accuracy with the 5 web image test set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The felt the first image (60 speed limit) might be hard because of the noise at the top of the image. But
somehow the model worked. Similarly I thought the left turn ahead might be harder as the sky color
matches quite a bit with the arrow background. But again the model seems to have worked.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)   						| 
| General caution     	| General caution 								|
| Keep right			| Keep right									|
| Turn left ahead	    | Turn left ahead					 		    |
| Speed limit (60km/h)	| Speed limit (60km/h)      					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Here are results (IMAGES, CLASSES and SOFTMAX probabilities) as seen in the 11th cell of the Ipython notebook:

##### IMAGE 1: web/web_3_speed30.jpeg

![alt text][image6] 

['Speed limit (30km/h)', 'Speed limit (20km/h)', 'Speed limit (50km/h)', 'Speed limit (70km/h)', 'End of speed limit (80km/h)']

[1.0000000e+00 2.5377219e-15 4.2675879e-17 8.4852536e-20 1.5857130e-20]

##### IMAGE 2: web/web_5_caution.jpeg

![alt text][image8]

['General caution', 'Traffic signals', 'Pedestrians', 'Right-of-way at the next intersection', 'Road narrows on the right']

[1.0000000e+00 3.7188491e-10 7.5343872e-12 1.1531469e-14 2.2062558e-17]

##### IMAGE 3: web/web_2_side_arrow.jpeg

![alt text][image5] 

['Keep right', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']

[1. 0. 0. 0. 0.]

##### IMAGE 4: web/web_4_turn_left.jpeg

![alt text][image7]

['Turn left ahead', 'Go straight or right', 'Ahead only', 'No passing', 'Keep right']

[9.9999976e-01 2.8315412e-07 2.0277779e-09 9.0749526e-12 7.5834764e-12]

##### IMAGE 5: web/web_1_speed60.jpeg

![alt text][image4]

['Speed limit (60km/h)', 'End of no passing by vehicles over 3.5 metric tons', 'Slippery road', 'Speed limit (80km/h)', 'No passing for vehicles over 3.5 metric tons']

[9.9549592e-01 4.4553182e-03 4.1600153e-05 5.0858407e-06 1.4576126e-06]