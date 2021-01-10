import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
x=tf.placeholder(shape=[1,3],dtype=tf.float32)

w=tf.Variable(tf.ones([3,3]),dtype=tf.float32)
b=tf.Variable(1,dtype=tf.float32)

y=tf.matmul(x,w)+b
sess=tf.Session()

sess.run(tf.global_variables_initializer())

sess.run(y,feed_dict={x:[[1,2,3]]})

writer=tf.summary.FileWriter("graph",sess.graph)