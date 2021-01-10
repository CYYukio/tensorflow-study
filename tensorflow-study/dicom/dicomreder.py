import pydicom
import pylab

filename = './000000.dcm'
ds = pydicom.dcmread(filename)
ds.dir()  # 查看病人所有信息字典keys
print(ds.PatientName)  # 查看病人名字
print(ds) # 查看病人所有信息字典， 如果出现某key对应值编码错误，先暂时跳过该key

############ 查看dicom对应图片值 #####################
print(ds.pixel_array.shape)
print(ds.pixel_array)
##读取显示图片
pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
pylab.show()

##修改图片中的元素，不能直接使用data_array,需要转换成PixelData
'''
for n,val in enumerate(ds.pixel_array.flat): # example: zero anything < 300
    if val < 300:
        ds.pixel_array.flat[n]=0
ds.PixelData = ds.pixel_array.tostring()
ds.save_as('newfilename.dcm')

'''
