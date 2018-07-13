
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

#a fat matrix
N = 50
D = 50 

X = (np.random.random((N,D)) - 0.5)*10 #random.random: Return random floats in the half-open interval [0.0, 1.0).
#X will have numbers between -5 and 5

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

costs = []
w = np.random.randn(D) / np.sqrt(D) #random initialization
learning_rate = 0.001
l1 = 10.0

# J = (Y - Xw)T . (Y - Xw)
# dJ/dw = -2X.T.Y + 2X.T.Xw = 2X.T(Yhat - Y)

for t in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w)) #The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

print("True w:", true_w)
print("Final w:", w)

plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()

