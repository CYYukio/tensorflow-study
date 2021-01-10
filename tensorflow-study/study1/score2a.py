import random
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


x=tf.placeholder(shape=[3],dtype=tf.float32)

#0.1是初始值
w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
wn=tf.nn.softmax(w)#softmax规范∑w=1

b=tf.Variable(80,dtype=tf.float32)

n1=x*wn
n2=tf.reduce_sum(n1)-b
y=tf.nn.sigmoid(n2)


#目标计算结果
goodstu=tf.placeholder(shape=[],dtype=tf.float32)

loss=tf.abs(y-goodstu)


#优化器,0.001是学习率
optimizer=tf.train.RMSPropOptimizer(0.1)
train=optimizer.minimize(loss)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

random.seed()
for i in range(500):
    xData = [int(random.random() * 30 + 70), int(random.random() * 30 + 70), int(random.random() * 30 + 70)]
    xAll = xData[0]*0.6+xData[1]*0.3+xData[2]*0.1
    if xAll>=85:
        Y=1
    else:
        Y=0

    result=sess.run([train,x,goodstu,wn,b,n2,y,loss],feed_dict={x:xData,goodstu:Y})

    print(result)