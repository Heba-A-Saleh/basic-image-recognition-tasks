# basic-image-recognition-processing

This repository contains a tasks to perform Image Recognition . The task involves performing basic image recognition tasks on any dataset. within these tasks the model were trained from a dataset that consists of 315 training images and 90 test images, each of size 120x128 pixels. Each image in the dataset has four characteristics: name, direction faced, emotion, and eyewear. The goal is to train models to predict the direction faced and emotion of the person in the image.

## Dataset Information

The dataset used in this project is confidential and cannot be shared publicly due to legal and ethical reasons. The dataset consists of images in JPEG format of dimensions 120x128, with the training set containing 315 images and the test set containing 90 images. Each image has four characteristics: name, direction faced, emotion, and eyewear, which are encoded in the image file name. The direction faced can be left, right, straight, or up, and the emotion can be happy, sad, neutral, or angry. The eyewear can be open or sunglasses. To replicate the experiment use a public dataset wiht similar features. 

I acknowledge that using a different dataset may have limitations and may affect the performance of the model. However, due to the confidentiality of the original dataset, we believe that this is the best alternative for replicating our experiment.

## Direction Faced Analysis

In this task, I create X_train using images in the TrainingSet folder. I use the PIL module to open JPEG files and convert them to grayscale. Each image has a shape of 120x128. I flatten each image array to a vector of dimensions 1x15360, and the label of the image is maintained from the file name. I create y_train_directionfaced using the images' file names, where the label of the image is encoded using the following dictionary:

```
direction_encode = {'right': 0, 'left': 1, 'up': 2, 'straight': 3}
```

I create X_test and y_test_directionfaced arrays using the TestSet folder. 

### First task train using a Random Forest classifier 
model will be modeled with parameters of random_state=0 on the training dataset and time how long it takes. We evaluate the resulting model on the test set and return the trained model, time of training, and accuracy on the test set in a pickle format as part_a.pkl.

###  Second task train a new Random Forest classifier on a reduced dataset
I used PCA to reduce the training dataset's dimensionality, with a variance ratio of 95%. I train a new Random Forest classifier on the reduced dataset.

## Emotional Analysis

In this section, I use emotion as a label. The label of each image is encoded using the following dictionary:

```
emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
```

I create y_train_emotion and t_test_emotion according to the emotion label.

### Third task train a Logistic Regression classifier 
The model will be trained with parameters of multi_class="multinomial", solver="lbfgs", random_state=0 on the training dataset 

### Forth task train a new Logistic Regression classifier on the reduced dataset
I used PCA to reduce the training dataset's dimensionality, with a variance ratio of 95%. I train a new Logistic Regression classifier on the reduced dataset 


For all tasks the trained model, time of training, and accuracy on the test set in a pickle format.
