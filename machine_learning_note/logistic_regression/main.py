import numpy as np, numpy.random as random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Preparing data
bc = datasets.load_breast_cancer()
X, y, feature_names, target_names = bc.data, bc.target, bc.feature_names, bc.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2910)

n_samples, n_features = X_train.shape

df_dict = {name:X_train[:, idx] for idx,name in enumerate(feature_names)}
df_dict["result"] = np.copy(y_train)
df = pd.DataFrame(df_dict)
print(f"\nBang du lieu: \n",df)




