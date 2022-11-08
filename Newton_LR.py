#!/usr/bin/env python
# coding: utf-8

# In[46]:


## Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing required libraries  
from sklearn.datasets import load_breast_cancer  
import pandas as pd 
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import KFold   
from sklearn.metrics import accuracy_score


# In[48]:


data = pd.read_csv('data.csv')

## Dropping a unused fields
fields_to_drop = ['id', 'Unnamed: 32'] 
data = data.drop(fields_to_drop, axis=1)

## Converting diagnosis to int - 1 for malignant, 0 - for benign
d = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(d)

## Visualising the data set
data.head()


# In[68]:


## Using 20% of dataset for testing
test_split_idx = int(data.shape[0]*0.2) 
val_split_idx = int(data.shape[0]*0.1) 

test_data = data[test_split_idx:]
val_data = data[val_split_idx:test_split_idx]
data = data[:val_split_idx]

## Separating data to features and targets
train_Y, train_X = data['diagnosis'], data.drop('diagnosis', axis=1)
val_Y, val_X = val_data['diagnosis'], val_data.drop('diagnosis', axis=1)
test_Y, test_X = test_data['diagnosis'], test_data.drop('diagnosis', axis=1)


# In[67]:


train_X.head()


# In[51]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[52]:


def newton_step(curr, y, X, reg=None):
    p = np.array(sigmoid(X.dot(curr[:,0])), ndmin=2).T  # probability matrix - N x 1
    W = np.diag((p*(1-p))[:,0]) # N by N diagonal matrix
    hessian = X.T.dot(W).dot(X)  # 30 by 30 matrix
    grad = X.T.dot(y-p)  # 30 by 1 matrix
    
    # regularization step
    if reg:
        step = np.dot(np.linalg.inv(hessian + reg*np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)
        
    beta = curr + step
    
    return beta


# In[53]:


def check_convergence(beta_old, beta_new, tol, iters):
    coef_change = np.abs(beta_old - beta_new)
    return not (np.any(coef_change>tol) and iters < max_iters)


# In[54]:


def test_model(X, y, beta):
    prob = np.array(sigmoid(X.dot(beta)))
    
    ## Converting prob to prediction, >.5 = True, <.5 = False
    prob = np.greater(prob, 0.5*np.ones((prob.shape[1],1)))
    accuracy = np.count_nonzero(np.equal(prob, y))/prob.shape[0] * 100
    return accuracy


# In[55]:


## Hyperparameters
max_iters = 20
tol=0.1 # convergence tolerance
reg_term = 1

beta_old, beta = np.ones((30,1)), np.zeros((30,1))
iter_count = 0
coefs_converged = False

while not coefs_converged:
    print('Iteration: {}'.format(iter_count))
    print('Validation Accuracy: {}%'.format(
        test_model(val_X, val_Y.to_frame(), beta_old)))
    beta_old = beta
    beta = newton_step(beta, train_Y.to_frame(), train_X, reg_term)
    iter_count += 1
    coefs_converged = check_convergence(beta_old, beta, tol, iter_count)


# In[56]:


print('After {} Iterations'.format(iter_count))
print('Test Accuracy: {}%'.format(
        test_model(test_X, test_Y.to_frame(), beta)))


# In[57]:


beta


# In[30]:


def gd_step(curr, y, X, lr=0.0000001):
    hx = X.dot(curr)
    p = np.array(sigmoid(hx))
    change = lr * (X.T.dot(y-p))
    beta = curr + change  
    
    return beta


# In[31]:


# Hyperparameters
batch_size = 50
lr = 0.0001
max_iters = 51

beta_old, beta = np.ones((30,1)), np.zeros((30,1))
iter_count = 0

while iter_count < max_iters:
    if iter_count % 10 == 0:
        print('Epoch: {}'.format(iter_count))
        print('Validation Accuracy: {}%'.format(
            test_model(val_X, val_Y.to_frame(), beta)))
    beta_old = beta
    for i in range(0, train_X.shape[0], batch_size):
        beta = gd_step(beta, train_Y[i:i+batch_size].to_frame(), 
                        train_X[i:i+batch_size], lr)
    iter_count += 1


# In[32]:


print('After {} Iterations'.format(iter_count))
print('Test Accuracy: {}%'.format(
        test_model(test_X, test_Y.to_frame(), beta)))


# In[33]:


beta


# In[42]:





# In[ ]:




