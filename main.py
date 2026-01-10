import numpy as np




# linear regression
class LinearRegression:
    def __init__(self):
        self._theta = None

    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        self._theta = value

    def predict(self, x):
            if self._theta is None:
                raise ValueError("you have not fitted model yet")
            return x.dot(self._theta)
    
    def fit_normal_equ(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        value = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self._theta = value

    def fit_svd(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        value = np.linalg.pinv(X_b).dot(y)
        self._theta = value
    




X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit_normal_equ(X, y)
# lin_reg.fit_svd(X, y)

# test
test_sample = np.random.rand(1, 1)
test_y = 3 + 4 * test_sample + np.random.randn(1, 1)
test_sample_b = np.c_[np.ones((1, 1)), test_sample]

print(test_y)
print(lin_reg.predict(test_sample_b))












