import numpy as np, numpy.random as random

class LogisticRegression:

    def __init__(self,learning_rate=0.001, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight =None
        self.bias = None
        self.activate_func = self.sigmoid_func
        
    # Trainging function
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # init weights
        self.weight = random.rand(n_features)
        self.bias = 1

        w_old = self.weight.copy()
        is_break_loop = False
        counter = 0
        # Training loop with SGD
        for i in range(self.n_iters):
            # shuffle data
            shuffle_idx = random.permutation(n_samples)
            for idx in shuffle_idx:
                xi = X[idx]
                yi = y[idx]
                # Calculate predicted value:
                linear_y = self.weight.T.dot(xi) + self.bias
                # Apply activate function
                y_pred = self.activate_func(linear_y)
                # Calculate loss grad
                dloss_dw = (y_pred - yi)*xi
                dloss_db = (y_pred - yi)

                # Stopping criteria
                self.weight -= self.learning_rate*dloss_dw
                if (counter+1) % 20 == 0:
                    print("count",counter)
                    if np.linalg.norm(self.weight - w_old)/len(self.weight) < 1e-3:
                        print(f"Converged after {counter} iters")
                        return
                self.bias -= self.learning_rate*dloss_db
                w_old = self.weight.copy()
            print(f"lan {counter + 1}")
            counter +=1
       



    def has_converged(self, loss):
        pass    

    # Predict new input
    def predict(self, X):
        linear_output = X.dot(self.weight) + self.bias
        y_pred = self.activate_func(linear_output)
        y_pred_cls = [1 if i >0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls) 

    # Activate functions 
    def sigmoid_func(self, X):
        return 1/(1+np.exp(-X))


if __name__ == "__main__":
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

    # Traing and predict step
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)
    y_pred = logistic_reg.predict(X_test)

    compare_df = {
        "pred":y_pred,
        "true":y_test
    }
    compare_df = pd.DataFrame(compare_df)
    compare_df.to_csv('./ok.csv')
    print(compare_df)
    print(f"Accuracy: { np.sum(y_pred == y_test)/len(y_pred)}")



