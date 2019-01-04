#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:36:51 2019

@author: nitin
"""
import tensorflow as tf

# print tensorflow version
print(tf.__version__)


hello = tf.constant("Hello ")
world = tf.constant("World")

type(hello)

result = hello + world

print(result)

type(result)

with tf.Session() as session:
    print(session.run(result))
    #r = session.run(result)
    #print(r)
 
# Computation
tensor_1 = tf.constant(1)
tensor_2 = tf.constant(2)

#print session
session


# Operations
const = tf.constant(10)

# 4 x 4 matrix with value 10
fill_mat = tf.fill((4,4),10)

myzeros = tf.zeros((4,4))

myones = tf.ones((4,4))

myrandn = tf.random_normal((4,4),mean=0,stddev=0.5)

myrandu = tf.random_uniform((4,4),minval=0,maxval=1)

my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

with tf.Session() as sess:
    for op in my_ops:
        print(op.eval())
        print('\n')
        
# Matrix multiplication
a = tf.constant([ [1,2],
                  [3,4]  ])
    
a.get_shape()

a.shape

b = tf.constant([[10], [20]])

b.get_shape()

result = tf.matmul(a,b)

with tf.Session() as session3:
    #print(session3.run(result))
    print(result.eval())

    
