import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''输入层'''
#placeholder,占位符，只有run的时候才会填入数据
x1 = tf.placeholder(dtype=tf.float32)
x2 = tf.placeholder(dtype=tf.float32)
x3 = tf.placeholder(dtype=tf.float32)

'''权重'''
#0.1是初始值
w1 = tf.Variable(0.1,dtype=tf.float32)
w2 = tf.Variable(0.1,dtype=tf.float32)
w3 = tf.Variable(0.1,dtype=tf.float32)

'''隐藏层'''
n1 = x1*w1
n2 = x2*w2
n3 = x3*w3

'''输出层'''
y = n1+n2+n3

sess = tf.Session()

#给所有可变参数初始化
init = tf.global_variables_initializer()
sess.run(init)

result = sess.run([x1,x2,x3,w1,w2,w3,y], feed_dict={x1:90,x2:80,x3:70})
print(result)