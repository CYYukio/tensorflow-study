import tensorflow as tf
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

random.seed()

row = 10
xData=np.full(shape=(row,3),fill_value=0,dtype=np.float32)
yLabel=np.full(shape=(row),fill_value=0,dtype=np.float32)

for i in range(row):
    xData[i][0]=int(random.random()*30)+70
    xData[i][1]=int(random.random()*30)+70
    xData[i][2]=int(random.random()*30)+70

    yData=xData[i][0]*0.6+xData[i][1]*0.3+xData[i][2]*0.1

    if yData > 85:
        yLabel[i]=1
    else:
        yLabel[i]=0

#print(xData,yLabel)

xTrain=tf.placeholder(shape=[3],dtype=tf.float32)
yTrain=tf.placeholder(shape=[],dtype=tf.float32)

w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
b=tf.Variable(80,dtype=tf.float32)
wn=tf.nn.softmax(w)

n1=wn*xTrain
n2=tf.reduce_sum(n1)-b
y=tf.sigmoid(n2)

loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(0.1)
train=optimizer.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    for j in range(row):
        result=sess.run([train,xTrain,yTrain,wn,y,loss],feed_dict={xTrain:xData[j],yTrain:yLabel[j]})

        print(result)

trainResultPath="./save/score"
print("saving....")
tf.train.Saver().save(sess,save_path=trainResultPath)