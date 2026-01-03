import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
##linear regression 


X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1) # so our thetas uqual 4 and 3
# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# plt.show()


X_b = np.c_[np.ones((100, 1)), X]
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) ##normal equatation
theta2 = np.linalg.pinv(X_b).dot(y) # SVD
print(theta, theta2)
# print(theta)
# X_new = np.array([0, 2])
# X_new_b = np.c_[np.ones((2,1)), X_new]

# y_predict = X_new_b.dot(theta)
# print(y_predict)

# y_predict = X_b.dot(theta) 
# print(mean_squared_error(y, y_predict))
# print(y[:5], y_predict[:5])