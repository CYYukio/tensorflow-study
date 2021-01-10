import keras as k
import random
import numpy as np

random.seed()

rowSize=4
rowCount=8192

xData=np.full((rowCount,rowSize),5,dtype=np.float32)
yLabel=np.full((rowCount,2),0,dtype=np.float32)

for i in range(rowCount):
    for j in range(rowSize):
        xData[i][j]=np.floor(random.random()*10)
        if xData[i][2]%2==0:
            yLabel[i][0]=0
            yLabel[i][1]=1
        else:
            yLabel[i][0]=1
            yLabel[i][1]=0


model=k.models.Sequential()

model.add(k.layers.Dense(32,input_dim=4,activation='tanh'))
model.add(k.layers.Dense(32,input_dim=32,activation='sigmoid'))
model.add(k.layers.Dense(2,input_dim=32,activation='softmax'))
model.compile(loss='mean_squared_error',optimizer="RMSProp",metrics=['accuracy'])
model.fit(xData,yLabel,epochs=10,batch_size=1,verbose=2)


xTest=np.array([[4,5,3,7],[2,1,2,6],[9,8,7,6],[0,1,9,3],[3,3,0,3]],dtype=np.float32)
for i in range(len(xTest)):
    resultAry=model.predict(np.reshape(xTest[i],(1,4)))
    print("x:%s ,y:%s" % (xTest[i],resultAry))