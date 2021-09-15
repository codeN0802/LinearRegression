from load_dataset import x_train, x_test, y_train, y_test, X,y
from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics


mlr = MultipleLinearRegression()

# fit our LR to our data
mlr.fit(x_train, y_train)
# make predictions and score
pred = mlr.predict(x_test)
print(mlr.coefficients)
print(mlr.intercept)
# calculate r2_score
score = mlr.r2_score(y_test, pred)
print(f'Our Final R^2 score: {score}')
print('OurMSE:', metrics.mean_squared_error(y_test, pred))
print('=================================================')


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
