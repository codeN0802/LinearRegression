from load_dataset import x_train, x_test, y_train, y_test, X,y
from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from PIL import Image
from numpy import array
import pandas as pd
import  glob
import  numpy as np
from os import listdir
from matplotlib import image
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
#
# # fit scikit-learn's LR to our data
# sk_mlr.fit(x_train, y_train)
# print(sk_mlr.coef_)
# print(sk_mlr.intercept_)
# # predicts and scores
# sk_score = sk_mlr.score(x_test, y_test)
# print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
# # Press the green button in the gutter to run the script.
#
# predictions = sk_mlr.predict(x_test)
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('Du doan:', sk_mlr.predict([[62,0,140,268,0,160,0,3.6,0,2]]))
from matplotlib.image import imread
img1 = Image.open('stare/1/im0007.jpg')
ar = np.array(img1)
print(ar)
# for row in ar:
#     outputstring=""
#     for column in row:
#         valuestring=""
#         for value in column:
#               if value < 10:
#                   valuestring+="00"+str(value)
#               elif value < 100:
#                   valuestring+="0"+str(value)
#               else:
#                   valuestring+=str(value)
#         outputstring+=valuestring+";"
#     print(outputstring[:-1]) #excluding the last letter, because its a ;
import csv

# def csvWriter(fil_name, nparray):
#   example = nparray.tolist()
#   with open(fil_name+'.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerows(example)
#
# csvWriter("myfilename", ar)

# print('========================================================')
#
# filenames = glob.glob('stare/1/*.jpg')
# images = [Image.open(fn).convert('L') for fn in filenames]
# data = np.stack([np.array(im) for im in images])
# print(data)

from matplotlib.pyplot import imread
import numpy as np
import pandas as pd
import os
# import imageio
import glob
import pathlib
# v = []
# for i,files in enumerate(pathlib.Path('./Stare/').glob('./**/*.jpg')):
#     im = imread(files.as_posix())
#     value = im.flatten()
#     value = np.hstack((int(files.parent.name),value))
#     v.append(value)
# df = pd.DataFrame(v)
# df = df.sample(frac=1)
# df.to_csv('files_path.csv',header=False,index=False)
