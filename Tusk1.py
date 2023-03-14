#!/usr/bin/env python
# coding: utf-8

# # Подберите скорость обучения() и количество итераций

# In[1]:


n = X.shape[1]  
alpha = 1e-04

W = np.array([1, 0.5])   
W, alpha   


# In[2]:


for i in range(1500):  
    y_pred = W @ X  
    err = calc_mse(y, y_pred)  
    for ii in range(W.shape[0]):  
        W[ii] -= alpha * (1 / n * 2 * np.sum(X[ii] * (y_pred - y)))  
    if i % 100 == 0:  
        print(i, W, err)  


# В этом коде мы избавляемся от итераций по весам, но тут есть ошибка, исправьте ее

# In[3]:


for i in range(287):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    # for ii in range(W.shape[0]):
    # W[ii] -= alpha * (1/n * 2 * np.sum(X[ii] * (y_pred - y)))
    W -= (alpha * (1 / n * 2 * np.sum(X * (y_pred - y))))
    if i % 10 == 0:
        print(i, W, err)


# In[ ]:




