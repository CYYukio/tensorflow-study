import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''输入层'''
#placeholder,占位符，只有run的时候才会填入数据
x=tf.placeholder(shape=[3],dtype=tf.float32)

#0.1是初始值
w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
wn=tf.nn.softmax(w)#softmax规范∑w=1

n=x*w

y=tf.reduce_sum(n)

#目标计算结果
realScore=tf.placeholder(shape=[],dtype=tf.float32)

loss=tf.abs(y-realScore)


#优化器,0.001是学习率
optimizer=tf.train.RMSPropOptimizer(0.001)
train=optimizer.minimize(loss)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


for i in range(200):
    result1=sess.run([train,x,w,wn,y,realScore,loss],feed_dict={x:[50,60,70],realScore:63})
    result2=sess.run([train,x,w,wn,y,realScore,loss],feed_dict={x:[90,97,88,],realScore:93})

print(result1)
print(result2)
