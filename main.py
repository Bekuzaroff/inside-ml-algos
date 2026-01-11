import numpy as np



# SGD
class SGDRegressor:
    def __init__(self, learning_rate, max_iters):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.theta = None

    def fit_batch(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        rows = X_b.shape[0]
        columns = X_b.shape[1]
        m = rows

        self.theta = np.random.randn(columns, 1)
        for _ in range(self.max_iters):
            gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta = self.theta - self.learning_rate * gradients
        
    def fit_stochastic(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_epochs = 50
        t0, t1 = 5, 50

        rows = X_b.shape[0]
        cols = X_b.shape[1]

        self.theta = np.random.randn(cols, 1)

        for _ in range(n_epochs):
             for i in range(rows): # going throw amount of samples
                  random_index = np.random.randint(rows)
                  xi = X_b[random_index:random_index+1]
                  yi = y[random_index:random_index+1]
                  gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                  self.learning_rate = self.learning_schedule(t0, t1, n_epochs * rows + i)
                  self.theta = self.theta - self.learning_rate * gradients
                  
        
    def fit_mini_batch(self, X, y, batch_size=5):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        rows = X_b.shape[0]
        columns = X_b.shape[1]
        
        self.theta = np.random.randn(columns, 1)
        
        for epoch in range(self.max_iters):
            indices = np.random.permutation(rows)  # Перемешиваем индексы
            
            for start_idx in range(0, rows, batch_size):
                end_idx = min(start_idx + batch_size, rows)
                batch_indices = indices[start_idx:end_idx]
                
                x_mb = X_b[batch_indices]
                y_mb = y[batch_indices]
                
                # Вычисляем градиент
                m_batch = x_mb.shape[0]
                gradients = 2/m_batch * x_mb.T.dot(x_mb.dot(self.theta) - y_mb)
                self.theta = self.theta - self.learning_rate * gradients
        

                   
                   
    def predict(self, x):
         return x.dot(self.theta)
    
    def learning_schedule(self, t0, t1, t):
         return t0 / (t1 + t)


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
    




# X = 2 * np.random.rand(100, 1)
# y = 3 + 4 * X + np.random.randn(100, 1)

# lin_reg = LinearRegression()
# lin_reg.fit_normal_equ(X, y)
# # lin_reg.fit_svd(X, y)

# # test
# test_sample = np.random.rand(1, 1)
# test_y = 3 + 4 * test_sample + np.random.randn(1, 1)
# test_sample_b = np.c_[np.ones((1, 1)), test_sample]

# print(test_y)
# print(lin_reg.predict(test_sample_b))

X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)

sgd = SGDRegressor(0.1, 1000)
# sgd.fit_batch(X, y)
sgd.fit_mini_batch(X, y)
# sgd.fit_stochastic(X, y)

# test
test_sample = np.random.rand(1, 1)
test_y = 3 + 4 * test_sample + np.random.randn(1, 1)
test_sample_b = np.c_[np.ones((1, 1)), test_sample]

print(test_y)
print(sgd.predict(test_sample_b))












