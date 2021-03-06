from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split

# def sklearn_to_df(data_loader):
#     X_data = data_loader.data
#     X_columns = data_loader.feature_names
#     X = pd.DataFrame(X_data, columns=X_columns)
#
#     y_data = data_loader.target
#     y = pd.Series(y_data, name='target')
#
#     return X, y
#
# X, y = sklearn_to_df(load_boston())
df = pd.read_csv("heart.csv")
y = df.target
X = df.drop(['target'],axis=1)
print(X)
# X1 = df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']]
# y1 = df[['target']]
# print(X1)
# print(y1)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X.values)
print(y.values)
print(X.T)
print(y.T)
print(X)