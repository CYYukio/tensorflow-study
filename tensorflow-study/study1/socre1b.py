import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''输入层'''
#placeholder,占位符，只有run的时候才会填入数据
x1=tf.placeholder(dtype=tf.float32)
x2=tf.placeholder(dtype=tf.float32)
x3=tf.placeholder(dtype=tf.float32)

#0.1是初始值
w1=tf.Variable(0.1,dtype=tf.float32)
w2=tf.Variable(0.1,dtype=tf.float32)
w3=tf.Variable(0.1,dtype=tf.float32)

n1=x1*w1
n2=x2*w2
n3=x3*w3

y=n1+n2+n3

#目标计算结果
realScore=tf.placeholder(dtype=tf.float32)

loss=tf.abs(y-realScore)


#优化器,0.001是学习率
optimizer=tf.train.RMSPropOptimizer(0.001)
train=optimizer.minimize(loss)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


for i in range(200):
    result1=sess.run([train,x1,x2,x3,w1,w2,w3,y,realScore,loss],feed_dict={x1:50,x2:60,x3:70,realScore:63})
    result2=sess.run([train,x1,x2,x3,w1,w2,w3,y,realScore,loss],feed_dict={x1:90,x2:97,x3:88,realScore:93})

print(result1)
print(result2)
