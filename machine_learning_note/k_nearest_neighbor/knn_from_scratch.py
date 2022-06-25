import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


# Preparing datasets

iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_iris, Y_iris, test_size=0.3, random_state=2910)


print(f"\nFirst 5 records of training set:\n {X_train[:5, :]}")
print(f"\nFirst 5 labels of training set:\n {Y_train[:5]}")

X0 = X_train[ Y_train == 0]
X1 = X_train[ Y_train == 1]
X2 = X_train[ Y_train == 2]



plt.plot(X0[:,2], X0[:, 3], "r^")
plt.plot(X1[:,2], X1[:, 3], "go")
plt.plot(X2[:,2], X2[:, 3], "bs")
plt.show()

class KNN:
    def __init__(self, X_train, Y_train, k=3):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k

    def euclid_dist(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, X):
            y_pred = [self._predict(x) for x in X]
            return np.array(y_pred)
        

    def _predict(self,x):
        # Calculate distance
        distances = [self.euclid_dist(x, x_train) for x_train in X_train]
        # Find K points nearest x and their labels
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        knn_labels = [self.Y_train[idx] for idx in k_nearest_neighbors]
        # Find most common points in which cluster
        return Counter(knn_labels).most_common(1)[0][0]
        

clf = KNN(X_train, Y_train, 3)
y_pred = clf.predict(X_test)

table_comp = {
    "Y predicted":y_pred,
    "Y true":Y_test
}
df = pd.DataFrame(table_comp)
print(df)

print(f"Accuracy: {np.sum(y_pred == Y_test)/len(Y_test)}")