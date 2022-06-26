import numpy as np, numpy.random as random

class PLA:

    def __init__(self,w,bias, learning_rate = 1, n_iters = 100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = w
        self.bias = bias
        self.activate_func = self._unit_step_func


    def fit(self, X_train, y_train):
        # print(self.bias, X_train)
        Xbar = np.concatenate((self.bias, X_train), axis=1).T
        n_features, n_samples =Xbar.shape

        counter = 0
        # Training loop
        while True:
            
            permute_idx = random.permutation(n_samples)
            for idx in range(n_samples):
                xi = Xbar[:, permute_idx[idx]]
                yi = y_train[permute_idx[idx]]
                y_sgn_pred = self.w.T.dot(xi)
                y_pred = self.activate_func(y_sgn_pred)[0]

                # update weight
                if y_pred != yi:
                    # print(self.w, xi)
                    self.w += self.learning_rate*yi*(xi.reshape(self.w.shape[0],-1))

            # Check stop condition
            isConverged = self.has_converged(y_train, self.activate_func(self.w.T.dot(Xbar)))
            if isConverged or counter == self.n_iters:
                print(f"Converged after {counter} iters")
                break
            counter +=1
        return self.w

    def has_converged(self, y, y_pred):
        return np.array_equal(y, y_pred)

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        bias = np.ones((n_samples, 1))
        Xbar = np.concatenate((bias, X_test), axis=1).T
        linear_output = self.w.T.dot(Xbar)
        return self.activate_func(linear_output)

    def sgn_func(self, x):
        return np.sign(x)
    
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X, y = datasets.make_blobs( n_samples=150, n_features=2, centers= 2, cluster_std=1.05, random_state=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    n_samples = X_train.shape[0]
    print(n_samples)

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    plt.plot(X0[:,0], X0[:, 1], 'r^')
    plt.plot(X1[:,0], X1[:, 1], 'bs')
    plt.show()

    w_init = random.randn(3, 1)
    bias = np.ones((n_samples, 1))


    pla = PLA(w_init, bias,1,1000)
    w = pla.fit(X_train, y_train)
    print(w)
    y_pred = pla.predict(X_test)[0]
    print("Accuracy",np.sum( y_pred == y_test)/len(y_pred))

