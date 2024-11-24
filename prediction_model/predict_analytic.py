import numpy as np
import os


data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalize', 'data'))
data_minmax_path = os.path.join(data_root, 'data_minmax.csv')
theta_analytic_path = os.path.join(data_root, 'theta_analytic.txt')

data = np.loadtxt(data_minmax_path, delimiter=',',skiprows=1)
X = data[:, :-1]
y = data[:, -1]

# столбец единиц
X = np.c_[np.ones(X.shape[0]), X]

# theta = (X^T * X)^(-1) * X^T * y

X_T = X.T
theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
np.savetxt(theta_analytic_path,theta)
print(theta)

