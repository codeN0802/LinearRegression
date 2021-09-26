from load_dataset import x_train, x_test, y_train, y_test, X,y
from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from PIL import Image
from numpy import array
import pandas as pd
import  glob
import  numpy as np
from matplotlib import image
import sys
import os
import csv
import cv2
# mlr = MultipleLinearRegression()
#
# # fit our LR to our data
# mlr.fit(x_train, y_train)
# # make predictions and score
# pred = mlr.predict(x_test)
# print(mlr.coefficients)
# print(mlr.intercept)
# # calculate r2_score
# score = mlr.r2_score(y_test, pred)
# print(f'Our Final R^2 score: {score}')
# print('OurMSE:', metrics.mean_squared_error(y_test, pred))
# print('=================================================')
#
#
sk_mlr = LinearRegression()

# fit scikit-learn's LR to our data
sk_mlr.fit(x_train, y_train)
print(sk_mlr.coef_)
print(sk_mlr.intercept_)
# predicts and scores
sk_score = sk_mlr.score(x_test, y_test)
print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
# Press the green button in the gutter to run the script.

predictions = sk_mlr.predict(x_test)
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('Du doan:', sk_mlr.predict([[62,0,140,268,0,160,0,3.6,0,2]]))

def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


mydir = 'stare_train'

# load the original image
myFileList = createFileList(mydir)

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    print(np.array(img_file))


# X = []
# y = []
#
# data_folder = "stare"
# for folder in os.listdir(data_folder):
#     curr_path = os.path.join(data_folder,folder)
#     for file in os.listdir(curr_path):
#         curr_file = os.path.join(curr_path,file)
#         image = cv2.imread(curr_file)
#         X.append(image)
#         y.append(folder)
# print(X)
# print(y)

# img1 = Image.open('stare/1/im0007.jpg')
# ar = np.array(img1)
# print(ar)
# print("=========================================")
# img2 = cv2.imread('stare/1/im0007.jpg')
# print(img2)