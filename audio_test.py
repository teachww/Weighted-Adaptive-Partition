
import numpy 
import scipy
import csv
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import xlwt
import xlrd
import pandas as pd



def free_energy(x,W,hbias,vbias):
  n=x.shape[0]
  free_energy=0
  for i in range(n):
    v_sample=x[i]   
    free_energy+=(-numpy.dot(v_sample,vbias)-numpy.sum(softplus(numpy.dot(v_sample,W)+numpy.transpose(hbias))))/n
  #print('e1=',numpy.dot(v_sample,vbias))
  #print('e2=',numpy.dot(v_sample,W)+hbias)
  return free_energy

  
def softplus(x):
  return numpy.log(1+numpy.exp(x))
  

def sigmoid(x):
  return 1/(1+numpy.exp(-x))

def hidden_layer(x,weights,vbias,hbias):
  hidden = sigmoid(numpy.dot(x, weights) + numpy.transpose(hbias))

  '''
  num=x.shape[0]
  hidden=numpy.zeros((num,300),dtype=numpy.float32)
  for i in range(num):
    x_audio=x[i]
    hidden[i]=sigmoid(numpy.dot(x_audio,weights)+numpy.transpose(hbias))
  '''

  return hidden


    
  
  

def save(data,path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h,l] = data.shape  # h为行数，l为列数
    for i in range(h): 
        for j in range(l):
            sheet1.write(i,j,data[i,j])
    f.save(path)
def excel1m(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    c1 = numpy.arange(0, nrows, 1)
    datamatrix = numpy.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        minVals = min(cols)
        maxVals = max(cols)
        cols1 = numpy.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        ranges = maxVals - minVals
        b = cols1 - minVals
        normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = normcols # 把数据进行存储
    return datamatrix

def excel2m(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    c1 = numpy.arange(0, nrows, 1)
    datamatrix = numpy.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        minVals = min(cols)
        maxVals = max(cols)
        cols1 = numpy.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        #ranges = maxVals - minVals
        #b = cols1 - minVals
        #normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = cols # 把数据进行存储
    return datamatrix
  
data = excel2m('1test_a.xlsx')
#data_test = excel1m('D:/coding/mypaper/audio_test.xlsx')
#save(data,'train.xlsx')
#print(data)

print(data)
weights=excel2m('3vector_W.xlsx')
visible_bias=excel2m('3vector_vb.xlsx')
hidden_bias=excel2m('3vector_hb.xlsx')

hidden=sigmoid(numpy.dot(data,weights))
print(hidden)
print(numpy.shape(hidden))

'''

##########################hidden layer###########################
hidden=hidden_layer(data,weights,visible_bias,hidden_bias)
hidden=pd.DataFrame(hidden)
writer=pd.ExcelWriter('1test_hidden_a.xlsx')
hidden.to_excel(writer,'Sheet1')
writer.save()
#################################################################



energy=free_energy(data,weights,hidden_bias,visible_bias)
print(energy)
'''
'''
#######################free energy###########################
energy=[]

for i in range(5,720):
  aa=data[i:(i+1),:]
  e=free_energy(aa,weights,hidden_bias,visible_bias)
  energy.append(e)




x1=numpy.arange(0,715,1)
y1=energy
plot=plt.plot(x1,y1,linewidth=2)
#plt.ylim(0,-10000000)
plt.xlabel('minibatches')
plt.ylabel('free_energy')
plt.axvline(x=140,ls=':',c='red')
plt.axvline(x=268,ls=':',c='red')
plt.axvline(x=330,ls=':',c='red')
plt.axvline(x=468,ls=':',c='red')
plt.axvline(x=594,ls=':',c='red')
plt.show()

################################################################
'''



'''

def free_energy(energy,n1,n2):
    e=energy[n1:n2]
    num=(n2-n1)
    energy=sum(e)/num
    return energy

'''

n1=0
delta_n=20

a1=data[0:20,:]
delta_e=[]
window_size=[]
for i in range(delta_n):
  delta_e.append(0)
  window_size.append(20)
energy1=free_energy(a1,weights,hidden_bias,visible_bias)

print(energy1)


for i in range(1000):
  n2=n1+delta_n
  b1=data[n2:(n2+delta_n),:]
  print('b1',type(b1))
  
  if numpy.sum(b1)==0:
    break
  
 
  energy2=free_energy(b1,weights,hidden_bias,visible_bias)

  print(energy2)

  delta_energy=numpy.abs((energy2-energy1)/-210)
  print('energy_change,',(energy2-energy1),'delta_energy',delta_energy)
  if delta_energy>0.08:
    delta_n=20
  else:
    delta_n=min(60,2*delta_n)
  for i in range(delta_n):
    delta_e.append(delta_energy)
    window_size.append(delta_n)

  energy1=energy2
  n1=n2
  print('delta_n',delta_n)



energy=[]

for i in range(0,2590):
  aa=data[i:(i+1),:]
  e=free_energy(aa,weights,hidden_bias,visible_bias)
  energy.append(e)

figsize=13,8
figure, ax=plt.subplots(figsize=figsize)
x1=numpy.arange(0,2590,1)
#plot=plt.plot(x1,y1,linewidth=4,c='blue')
#plot=plt.plot(x1,y2,linewidth=4,c='blue',label='video input')
plot=plt.plot(x1,energy,linewidth=4)
#plot=plt.plot(x1,y4,linewidth=3,c='c',label='m3')

plt.xlabel('minibatches',fontproperties='Times New Roman',fontsize=32)
plt.ylabel('free_energy',fontproperties='Times New Roman',fontsize=32)
plt.axvline(x=200,ls=':',c='red',linewidth=3)
plt.axvline(x=450,ls=':',c='red',linewidth=3)
plt.axvline(x=550,ls=':',c='red',linewidth=3)
plt.axvline(x=750,ls=':',c='red',linewidth=3)
plt.axvline(x=900,ls=':',c='red',linewidth=3)

plt.axvline(x=1300,ls=':',c='red',linewidth=3)
plt.axvline(x=1450,ls=':',c='red',linewidth=3)
plt.axvline(x=1750,ls=':',c='red',linewidth=3)
plt.axvline(x=1850,ls=':',c='red',linewidth=3)
plt.axvline(x=2090,ls=':',c='red',linewidth=3)
plt.axvline(x=2290,ls=':',c='red',linewidth=3)

plt.yticks(size=26)
plt.xticks(size=26)

labels = ax.get_xticklabels()+ ax.get_yticklabels()
[label.set_fontname('Times New Roman')for label in labels]
#plt.legend(prop={'family':'Times New Roman','size':19})
figure.tight_layout()
plt.show()






figsize=13,8
figure, ax=plt.subplots(figsize=figsize)
x1=numpy.arange(0,len(delta_e),1)
#plot=plt.plot(x1,y1,linewidth=4,c='blue')
#plot=plt.plot(x1,y2,linewidth=4,c='blue',label='video input')
plot=plt.plot(x1,delta_e,linewidth=4)
#plot=plt.plot(x1,y4,linewidth=3,c='c',label='m3')
plt.ylim(0,0.5)
plt.xlabel('minibatches',fontproperties='Times New Roman',fontsize=32)
plt.ylabel('free_energy',fontproperties='Times New Roman',fontsize=32)
plt.axvline(x=200,ls=':',c='red',linewidth=3)
plt.axvline(x=450,ls=':',c='red',linewidth=3)
plt.axvline(x=550,ls=':',c='red',linewidth=3)
plt.axvline(x=750,ls=':',c='red',linewidth=3)
plt.axvline(x=900,ls=':',c='red',linewidth=3)

plt.axvline(x=1300,ls=':',c='red',linewidth=3)
plt.axvline(x=1450,ls=':',c='red',linewidth=3)
plt.axvline(x=1750,ls=':',c='red',linewidth=3)
plt.axvline(x=1850,ls=':',c='red',linewidth=3)
plt.axvline(x=2090,ls=':',c='red',linewidth=3)
plt.axvline(x=2290,ls=':',c='red',linewidth=3)
plt.axhline(y=0.08,ls='-',c='green')

plt.yticks(size=26)
plt.xticks(size=26)
labels = ax.get_xticklabels()+ ax.get_yticklabels()
[label.set_fontname('Times New Roman')for label in labels]
#plt.legend(prop={'family':'Times New Roman','size':19})
figure.tight_layout()
plt.show()





figsize=13,8
figure, ax=plt.subplots(figsize=figsize)
x1=numpy.arange(0,len(window_size),1)
#plot=plt.plot(x1,y1,linewidth=4,c='blue')
#plot=plt.plot(x1,y2,linewidth=4,c='blue',label='video input')
plot=plt.plot(x1,window_size,linewidth=4)
#plot=plt.plot(x1,y4,linewidth=3,c='c',label='m3')
plt.ylim(0,100)
plt.xlabel('minibatches',fontproperties='Times New Roman',fontsize=32)
plt.ylabel('free_energy',fontproperties='Times New Roman',fontsize=32)
plt.axvline(x=200,ls=':',c='red',linewidth=3)
plt.axvline(x=450,ls=':',c='red',linewidth=3)
plt.axvline(x=550,ls=':',c='red',linewidth=3)
plt.axvline(x=750,ls=':',c='red',linewidth=3)
plt.axvline(x=900,ls=':',c='red',linewidth=3)

plt.axvline(x=1300,ls=':',c='red',linewidth=3)
plt.axvline(x=1450,ls=':',c='red',linewidth=3)
plt.axvline(x=1750,ls=':',c='red',linewidth=3)
plt.axvline(x=1850,ls=':',c='red',linewidth=3)
plt.axvline(x=2090,ls=':',c='red',linewidth=3)
plt.axvline(x=2290,ls=':',c='red',linewidth=3)

plt.yticks(size=26)
plt.xticks(size=26)
labels = ax.get_xticklabels()+ ax.get_yticklabels()
[label.set_fontname('Times New Roman')for label in labels]
#plt.legend(prop={'family':'Times New Roman','size':19})
figure.tight_layout()
plt.show()







