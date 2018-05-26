import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0, 10, N) # Return evenly spaced numbers over a specified interval.
Y = 0.5*X + np.random.randn(N) #randn: Return a sample (or samples) from the “standard normal” distribution.

Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

X = np.vstack([np.ones(N), X]).T #adding the bias term

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y)) #eye: Return a 2-D array with ones on the diagonal and zeros elsewhere. - This is used to generate the identity matrix
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()