#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:47:01 2019

@author: nitin
"""
import tensorflow as tf
import numpy as np

# Set some random data
rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

# Tensofflow placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Operations
add_op = a + b # tf.add(a,b)
mult_op = a * b #tf.multiply(a,b)  <-- element wise multiplication
matrix_mul = tf.matmul(a,b) # <-- matrix multiplication

# Running Sessions to create Graphs with Feed Dictionaries
with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    
    print('\n')
    
    mult_result = sess.run(mult_op,feed_dict={a:rand_a,b:rand_b})
    print(mult_result)
    
    print('\n')
    
    matmul_result = sess.run(matrix_mul,feed_dict={a:rand_a,b:rand_b})
    print(matmul_result)    


