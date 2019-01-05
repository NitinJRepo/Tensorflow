#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:28:42 2019

@author: nitin
"""
import tensorflow as tf

my_tensor = tf.random_uniform((4,4),0,1)

# Tensorflow variable
my_variable = tf.Variable(initial_value = my_tensor)

print(my_variable)

# You need to "initialize" the variables first
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    #print(my_variable.eval())
    
    print(session.run(my_variable))

#======================================================
    
# Tensorflow placeholder
ph = tf.placeholder(tf.float64)

print(ph)

# For shape its common to use (None, # of Features) 
# Because None can be filled by number of samples in data
ph = tf.placeholder(tf.float32,shape=(None,5))

print(ph)