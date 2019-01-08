#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 18:34:53 2019

@author: nitin
"""
"""
Created on Sun Nov 25 13:03:44 2018

@author: nitin
"""
# Linear regression with tensorflow 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# load the data
dataset = pd.read_csv("./data/Housing.csv")

# Normalisisng the data
dataset = (dataset - dataset.mean())/dataset.std()
dataset.head()

X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

# Placeholders
xph = tf.placeholder(tf.float32)
yph = tf.placeholder(tf.float32)

m = tf.Variable(1.5)
b = tf.Variable(0.9)

# Graph
Yhat = (m * xph) + b

# Cost function for Linear Regression
error = tf.reduce_sum(tf.square(yph - Yhat))   #sum of squared error

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize tensorflow variables
init = tf.global_variables_initializer()

with tf.Session() as sess:  
    sess.run(init)
    
    iteration = 1000
    batch_size = 10
    
    for i in range(iteration):       
        random_index = np.random.randint(len(X), size=batch_size)       
        feed = {xph:X[random_index], yph:Y[random_index]}       
        sess.run(train, feed_dict = feed)
        
    model_m,model_b = sess.run([m,b])

# Calculate Yhat with final values of m and b
Yhat = X * model_m + model_b

plt.scatter(X, Y) # Plotting scatter points
plt.plot(X, Yhat,'r') # Plotting the line
plt.show()


