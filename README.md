# **Behavioral Cloning Project Writeup**


---

**Behavioral Cloning Project**

![Results][image7]

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUpImages/TraingLossVSValidationLoss.PNG "Model Visualization"
[image2]: ./WriteUpImages/cnn-architecture-624x890.png "Nvidia Model"
[image3]: ./WriteUpImages/CenterDriving.png "CenterDriving"
[image4]: ./WriteUpImages/Recovery.png "Recovery Image"
[image5]: ./WriteUpImages/Recovery2.png "Recovery Image"
[image6]: ./WriteUpImages/RightSideDriving.PNG "Normal Image"

[image7]: ./WriteUpImages/Demo.gif "Normal Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode ( I added the preprocessing part to fit with the model, from line #25 to #29 and #68 to #72)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 and track2.mp4 videos show the self-driving on track 1 and track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_50Iter_3rdRun.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (clone.py lines 201-205) 

The model includes RELU layers to introduce nonlinearity (code line 201 to line 210), and the data is normalized in the model using a Keras lambda layer (code line 199). 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 206). The model includes the dropout layer between the covolution and flatten layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 220). The preprocessing, augument, and shuffle functions are used to avoid overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The training model could run on both tracks without any running out or "accident". The car could run full speed on the track 1. The performance is even better than my driving performance.

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate of 0.0001, beta_1 of 0.9, and beta_2 of 0.999(clone.py line 213).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, I drove the car in the reversed direction, in order to compensate the steering angle bias on the both tracks. On the first track, I intentionally steering the angle left and right a little in order to add more adjustment stimulus and recovery training on the model. On the second track, I managed to make the speed to be constant although the road is up and down, and I tried hard to drive the car on the right side of the road. In this way, the car won't be confused by the data where the car cross the lane, and no recovery happens. Also, for one critical point on the first track, where after the bridge and no wall for a big turn, I drove the car back and forth several time to reinforce the training on that turn. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to 
(1) preprocess driving camera images to emphysize the road features in order to have better connection between the road condition to the steering angle. 

(2) Build the model based on paper from Nvidia's self-driving car model. Input is the preprocessed image, the output is the signal steering angle value for each image. 

(3) Capturing the driving data with good driving behavior, such as keeping the speed consistent, using continiouse steering instead of pulses of steering, look ahead for changing direction, and intentionally introducing some direction recovery on the road. 

(4) Train the model, find the place where the self-driving cause issue, collect more data around that spots, and train the model again. 

(5) Recording the fantastic driving (even better myself selfing) by trained AI with excitement, share with my friends.

My first step was to use a convolution neural network model similar to the Nvidia's self-driving car model. I thought this model might be appropriate because the paper showed the success on the self-driving. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used sklearn's train_test_split function to split the data into 80% for training and 20% for testing.

I didn't see this overfitting in my model that a low mean squared error on the training set but a high mean squared error on the validation set. The result on the validate on each epoch, I saw the validating loss is always lower than the traing loss. I think it is because I have large amount of data set, and the randomization as well as drop out are also helpful.

![alt text][image1]

To combat the overfitting, I modified the model so that I have dropout after the convolution layers, also, I added the random augumentation for the input image, such as flipping the image left to right, and changing the color of the image.

Then I tried with 10 epoch first, the model could handle the first track pretty well, I could use the maximum speed for let the model driving forever. 

But for the second track, 10 epoches with limited number of training seems not enough. Then, I drove the second track with three more cycles in one direction, and drove three more cycles in the other directions. This time, I tried with 50 epoches. The model worked great! It even learns to driving on the right side of the track!

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: in the first track, there is a spot where after the bridge, there is no wall to show the large/sharp turn. My model failed there, I just train it more with repeat that turns back and forth with several repeats. In the second track, there is a S shaped turn, the model passed the first track failed there. This time, I just add three more cycles for each direction, since I realize the data size is large, with several repeats on one spot may not be that helpful. Also, during the driving recording ,I intentionally driving the car left and right, in order to create some situation that need to recovery from the tilt angle. The model seems copied that behaviour showing a little Zigzag driving style on the straight road, haha!

At the end of the process, the vehicle drive full speed on the first track with no leaving from the center. And the vechicle drove much more smoothly on the second track than myself, I guess it is due to the speed control. The manual mode does not have the cruise control on the speed!

#### 2. Final Model Architecture

The final model architecture (clone.py lines 199-224) consisted of a convolution neural network with the following layers and layer sizes. According to the Nvidia's model, the input size of the model is 200x66 with YUV color, also, the pixel value is normalized. Therefore, I did the re-size and color converting in the preprocess. (I spent two days to find the difference of BGR and RGB between cv2.imread I used and Image.open drive.py used!!!)

The following five layers are 2D convolution layers the same as the Nvidia model. After the 2D convolution layers, I added a dropout layer. And then flatten, then fully-connected to converge to 1 output value.


The following is a visualization of the architecture (**Please note that there is a dropout layer between the last Convolution 64@1x18 and Flatten**)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image4]

![alt text][image5]

Also, I drove the car in the other direction in order to compensate the steering angle bias.

Then I repeated this process on track two in order to get more data points. It is very important to drive the car with constant speed, and steer the direction continiously. At the straight road, I would steer the direction right and left so that it could cause more steering changes, and "stimulus" the neural more. On the second track, I try my best to drive the car on the right side of the road, and the trained model learns this pretty well. 

![alt text][image6]

Also, I set the program running speed to fastest, so that the image size is small and data colletion/image process speed are fast. After the collection process, I had more than  80000 number of data points (including the right and left angle images).

For the preprocess,  first to cut the image to lower part of the view, since the road condition info in this part of image, and then, resize the image to the 66x200 being the same size of the input of the model. I convert the image from BGR (if using CV2.imread) to YUV). It is very important to check the color domain of the image read function, I had spent two days on finding the discrepency of image read functions between CV2.imread in clone.py and imageopen in drive.py. Therefore, I was wondering why the validation loss was just 0.03 but my car was driving like crazy! After fixing this color converting issue, my model could easily pass the first track!!!

To augment the data sat, I also flipped images and angles sincethis would prevent the overfitting issue, and making the training more efficient. The code the flip the image could be find in the augument function from line# 33 to line # 35. Also, I randomly add the color changes and image/steering changes to the training images. Those augumentations with the dropout and large training set, the overfitting issue could be solved. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used GPU for the Tensoflow training, 50 epoch with 50000 training data & 10000 validation data would not be very long time, 69s per epoch. As checked the training loss log, I would say 20 - 30 epoches would be enough. I used an adam optimizer, the learning rate is 0.0001, and with beta_1 of 0.9, and beta_2 of 0.999. After the training, the final loss is 0.0316.

The following video showing the car driving at the full speed on the first track, and driving with 15 Mph on the second track almost forever with no "accident".

[![Self Driving Behavior Cloning Track 1](http://img.youtube.com/vi/cWbxyaj9gsc/0.jpg)](http://www.youtube.com/watch?v=cWbxyaj9gsc "Self Driving Behavior Cloning Track 1")

[![Self Driving Behavior Cloning Track 2](http://img.youtube.com/vi/14vTI4GV_Wg/0.jpg)](http://www.youtube.com/watch?v=14vTI4GV_Wg "Self Driving Behavior Cloning Track 2")

The model could successfully run on both tracks. The first track with full speed, and the second track with 15 MPH. It is a really fun and rewarding project! I really enjoyed the skills I learn, and the final results I got for this project.