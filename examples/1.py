import tensorflow as tf
L = tf.constant(10)
B = tf.constant(20)
H = tf.constant(30)
V = L*B*H
A = 2 * (L*B+B*H+H*L)
