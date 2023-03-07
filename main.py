# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:35:23 2018

@author: Musaoglu
"""

from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import glob
import os
import time
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score



# TainingSet and TestData Creation - Face Direction
X_train = []
y_train_directionfaced = []

for train in glob.glob("C:/Users/Musaoglu/Dropbox/EECS 461/Assignments/A4/ML_Assignment4/TrainingSet/*.jpg"):
    path_train = os.path.splitext(train)[0]
    if 'left' in path_train:
        X_train.append(np.array(Image.open(train).convert('L')).flatten())
        y_train_directionfaced.append(1)
    elif 'right' in path_train:
        X_train.append(np.array(Image.open(train).convert('L')).flatten())
        y_train_directionfaced.append(0)
    elif 'up' in path_train:
        X_train.append(np.array(Image.open(train).convert('L')).flatten())
        y_train_directionfaced.append(2)
    elif 'straight' in path_train:
        X_train.append(np.array(Image.open(train).convert('L')).flatten())
        y_train_directionfaced.append(3)

X_train = np.array(X_train)
y_train_directionfaced = np.array(y_train_directionfaced)


X_test = []
y_test_directionfaced = []

for test in glob.glob("C:/Users/Musaoglu/Dropbox/EECS 461/Assignments/A4/ML_Assignment4/TestSet/*.jpg"):
    path_test = os.path.splitext(test)[0]
    if 'left' in path_test:
        X_test.append(np.array(Image.open(test).convert('L')).flatten())
        y_test_directionfaced.append(1)
    elif 'right' in path_test:
        X_test.append(np.array(Image.open(test).convert('L')).flatten())
        y_test_directionfaced.append(0)
    elif 'up' in path_test:
        X_test.append(np.array(Image.open(test).convert('L')).flatten())
        y_test_directionfaced.append(2)
    elif 'straight' in path_test:
        X_test.append(np.array(Image.open(test).convert('L')).flatten())
        y_test_directionfaced.append(3)
X_test = np.array(X_test)


# PART A
start_time = time.clock()
RF_Classifier_PartA = RandomForestClassifier(random_state=0)
RF_Classifier_PartA.fit(X_train, y_train_directionfaced)
end_time = time.clock()
execution_time = end_time - start_time

RF_predict_PartA = RF_Classifier_PartA.predict(X_test)
PartA_Accuracy = RF_Classifier_PartA.__class__.__name__, accuracy_score(y_test_directionfaced, RF_predict_PartA)
Part_A_Data = [RF_Classifier_PartA, execution_time, PartA_Accuracy[1]]
joblib.dump(Part_A_Data, 'part_a.pkl', protocol = 2)


# Part B
PCA_PartB = PCA(0.95)
PCA_PartB.fit(X_train)
Reduced_X_train = PCA_PartB.transform(X_train)
Reduced_X_test = PCA_PartB.transform(X_test)


start_time_PartB = time.clock()
RF_Classifier_PartB = RandomForestClassifier(random_state=0)
RF_Classifier_PartB.fit(Reduced_X_train, y_train_directionfaced)
end_time_PartB = time.clock()
execution_time_PartB= end_time_PartB - start_time_PartB

RF_predict_PartB = RF_Classifier_PartB.predict(Reduced_X_test)
PartB_Accuracy = RF_Classifier_PartB.__class__.__name__, accuracy_score(y_test_directionfaced, RF_predict_PartB)
Part_B_Data = [RF_Classifier_PartB, execution_time_PartB, PartB_Accuracy[1]]
joblib.dump(Part_B_Data, 'part_b.pkl', protocol = 2)


# Emotion Analysis
# TainingSet and TestData Creation 
X_train = []
y_train_emotion = []

for train_emotion in glob.glob("C:/Users/Musaoglu/Dropbox/EECS 461/Assignments/A4/ML_Assignment4/TrainingSet/*.jpg"):
    path_train_emotion = os.path.splitext(train_emotion)[0]
    if 'happy' in path_train_emotion:
        X_train.append(np.array(Image.open(train_emotion).convert('L')).flatten())
        y_train_emotion.append(1)
    elif 'neutral' in path_train_emotion:
        X_train.append(np.array(Image.open(train_emotion).convert('L')).flatten())
        y_train_emotion.append(0)
    elif 'sad' in path_train_emotion:
        X_train.append(np.array(Image.open(train_emotion).convert('L')).flatten())
        y_train_emotion.append(3)
    elif 'angry' in path_train_emotion:
        X_train.append(np.array(Image.open(train_emotion).convert('L')).flatten())
        y_train_emotion.append(2)

X_train = np.array(X_train)
y_train_emotion = np.array(y_train_emotion)

X_test = []
t_test_emotion = []

for test_emotion in glob.glob("C:/Users/Musaoglu/Dropbox/EECS 461/Assignments/A4/ML_Assignment4/TestSet/*.jpg"):
    path_test_emotion = os.path.splitext(test_emotion)[0]
    if 'happy' in path_test_emotion:
        X_test.append(np.array(Image.open(test_emotion).convert('L')).flatten())
        t_test_emotion.append(1)
    elif 'neutral' in path_test_emotion:
        X_test.append(np.array(Image.open(test_emotion).convert('L')).flatten())
        t_test_emotion.append(0)
    elif 'sad' in path_test_emotion:
        X_test.append(np.array(Image.open(test_emotion).convert('L')).flatten())
        t_test_emotion.append(3)
    elif 'angry' in path_test_emotion:
        X_test.append(np.array(Image.open(test_emotion).convert('L')).flatten())
        t_test_emotion.append(2)
X_test = np.array(X_test)

# PART C
start_time_PartC = time.clock()
LR_Classifier_PartC = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0)
LR_Classifier_PartC.fit(X_train, y_train_emotion)
end_time_PartC = time.clock()
execution_time_PartC = end_time_PartC - start_time_PartC

LR_predict_PartC = LR_Classifier_PartC.predict(X_test)
PartC_Accuracy = LR_Classifier_PartC.__class__.__name__, accuracy_score(t_test_emotion, LR_predict_PartC)
Part_C_Data = [LR_Classifier_PartC, execution_time_PartC, PartC_Accuracy[1]]
joblib.dump(Part_C_Data, 'part_c.pkl',  protocol = 2)

# Part D
PCA_PartD = PCA(0.95)
PCA_PartD.fit(X_train)
Reduced_X_train_Emotions = PCA_PartD.transform(X_train)
Reduced_X_test_Emotions = PCA_PartD.transform(X_test)


start_time_PartD = time.clock()
LR_Classifier_PartD = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0)
LR_Classifier_PartD.fit(Reduced_X_train_Emotions, y_train_emotion)
end_time_PartD = time.clock()
execution_time_PartD = end_time_PartD - start_time_PartD

LR_predict_PartD = LR_Classifier_PartD.predict(Reduced_X_test_Emotions)
PartD_Accuracy = LR_Classifier_PartD.__class__.__name__, accuracy_score(t_test_emotion, LR_predict_PartD)
Part_D_Data = [LR_Classifier_PartD, execution_time_PartD, PartD_Accuracy[1]]
joblib.dump(Part_D_Data, 'part_d.pkl',  protocol = 2)


a = joblib.load('part_a.pkl')
b = joblib.load('part_b.pkl')
c = joblib.load('part_c.pkl')
d = joblib.load('part_d.pkl')