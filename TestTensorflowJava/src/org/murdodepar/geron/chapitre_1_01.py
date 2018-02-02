import numpy as np
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)
X_b =  np.c_[np.ones((100, 1)), X] #ajouter x0 = 1 à chaque obs
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b).dot(Y)
