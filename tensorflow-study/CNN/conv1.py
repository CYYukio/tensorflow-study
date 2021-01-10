import numpy as np

xData1=np.array([[0,0],[0,1]],dtype=np.float32)
xData2=np.array([[0,1],[1,0]],dtype=np.float32)

filterT=np.array([[1,2],[3,4]],dtype=np.float32)

y1=np.sum(xData1*filterT)
y2=np.sum(xData2*filterT)
print(y1,y2)