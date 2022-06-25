import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
# X data features: chieu dai dai hoa, chieu rong dai hoa , chieu dai canh hoa, chieu rong canh hoa
iris_X = iris.data
iris_y = iris.target

print(f"Number of classes: {len(np.unique(iris_y))}")
print(f"Number of data points: {len(iris_y)}")

X0 = iris_X[iris_y == 0, :]
X1 = iris_X[iris_y == 1, :]
X2 = iris_X[iris_y == 2, :]


print(f"\nFirst 5 samples from class 0:\n {X0[:5, :]}")
print(f"\nFirst 5 samples from class 1:\n {X1[:5, :]}")
print(f"\nFirst 5 samples from class 2:\n {X2[:5, :]}")

X_train , X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

