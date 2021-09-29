# from load_dataset import x_train, x_test, y_train, y_test
from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
# sk_mlr = LinearRegression()
# sk_mlr.fit(x_train,y_train)
# print(sk_mlr.coef_)
# print(sk_mlr.intercept_)
#
# sk_score = sk_mlr.score(x_test, y_test)
# print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
# # # Press the green button in the gutter to run the script.
# #
# predictions = sk_mlr.predict(x_test)
# print('MSE:', metrics.mean_squared_error(y_test, predictions))






# sk_mlr = LinearRegression()
#
# # fit scikit-learn's LR to our data
# sk_mlr.fit(X_train,y_train)
# print(sk_mlr.coef_)
# print(sk_mlr.intercept_)
# img = Image.open('stare_test/0_test/AnyConv.com__im0297.jpg')
# a = np.array(img)
# b = np.transpose(a).reshape(a.shape[0_n], a.shape[1_n] * a.shape[2])
# c = b.flatten()
# print('Du doan:', sk_mlr.predict([c]))
# with open("abc.csv", "a") as f :
#     writer = csv.writer(f)
#     writer.writerow(sk_mlr.coef_)
# # predicts and scores
# # sk_score = sk_mlr.score(x_test, y_test)
# # print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
# # Press the green button in the gutter to run the script.
#
# # predictions = sk_mlr.predict(x_test)
# # print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('Du doan:', sk_mlr.predict([[62,0_n,140,268,0_n,160,0_n,3.6,0_n,2]]))

# x_train = []
# y_train = []
# x_test =[]
# y_test=[]
X=[]
y=[]
def createFileList(myDir, format='.jpg'):
    fileList = []
    # print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


# mydir = 'stare_train'
# mydirTest = 'stare_test'
mydirN = 'stare'
#
myFileListN = createFileList(mydirN)
for file in myFileListN:
    if file.__contains__("0_n"):
        y.append(0)
    if file.__contains__("1_n"):
        y.append(1)
    img_file = Image.open(file)
    a = np.array(img_file)
    b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
    c = b.flatten()

    X.append(c)
    Xn=np.array(X)

    Yn=np.array(y)
print(Xn)

print(Yn)
X_train , X_test , y_train, y_test = train_test_split(Xn,Yn,test_size=0.3,random_state=42)
# myFileList = createFileList(mydir)
# myFileListTest = createFileList(mydirTest)
# for file in myFileList:
#     if file.__contains__("0_train"):
#         y_train.append(0)
#     if file.__contains__("1_train"):
#         y_train.append(1)
#     img_file = Image.open(file)
#     a = np.array(img_file)
#     b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
#     c = b.flatten()
#     x_train.append(c)
#     x_train1=np.array(x_train)
#     y_train1=np.array(y_train)
#
# print(x_train)
# print(y_train)
# print(x_train1)
# print(y_train1)
# print("===================")
# for file in myFileListTest:
#     if file.__contains__("0_test"):
#         y_test.append(0)
#     if file.__contains__("1_test"):
#         y_test.append(1)
#     img_file = Image.open(file)
#     a = np.array(img_file)
#     b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
#     c = b.flatten()
#     x_test.append(c)
#     x_test1 = np.array(x_test)
#     y_test1 = np.array(y_test)
# print(x_test)
# print(y_test)
# print(x_test1)
# print(y_test1)
#
#
sk_mlr = LinearRegression()
sk_mlr.fit(X_train,y_train)
print(sk_mlr.coef_)
print(sk_mlr.intercept_)
# #
sk_score = sk_mlr.score(X_test, y_test)
print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
predictions = sk_mlr.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, predictions))
# img = Image.open('stare/1_n/im0399.jpg')
# a = np.array(img)
# b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
# c = b.flatten()
# print(c)
# print('Du doan:', sk_mlr.predict([c]))