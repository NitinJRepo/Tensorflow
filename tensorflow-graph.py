#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:21:32 2019

@author: nitin
"""

import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2

# Using with auto-closes the session
with tf.Session() as session:
    result = session.run(n3)
print(result)

# Get default graph
print(tf.get_default_graph())


# Get new graph
g = tf.Graph()

print(g)

# Setting a graph as default
graph_one = tf.get_default_graph()
graph_two = tf.Graph()

print(graph_one is tf.get_default_graph())

print(graph_two is tf.get_default_graph())


with graph_two.as_default():
    print(graph_two is tf.get_default_graph())
    
print(graph_two is tf.get_default_graph())