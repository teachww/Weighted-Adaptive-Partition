from __future__ import print_function
import numpy as np
import scipy
import csv
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import xlwt
import xlrd
import pandas as pd
#import study


  
def free_energy(x,w_u,w_v,visible_bias,hidden_bias):
    num=x.shape[0]
    energy=np.zeros([1])
    for i in range(num):
        x_image=x[i]   ##75*50
        
        #energy+= -np.sum(np.matmul(np.transpose(x_image),visible_bias)) - np.sum(softplus(np.matmul(np.matmul(w_u, x_image), np.transpose(w_v)) + hidden_bias))
        energy+=- np.trace(np.matmul(np.transpose(x_image),visible_bias)) - np.sum(softplus(np.trace(np.matmul(np.matmul(w_u, x_image), np.transpose(w_v)) + hidden_bias)))
        #print('e1=',np.matmul(np.matmul(w_u, x_image), np.transpose(w_v)) + hidden_bias)
        #print('e2=',np.sum(softplus(np.trace(np.matmul(np.matmul(w_u, x_image), np.transpose(w_v)) + hidden_bias))))
    energy=energy/num
    
    #print('energy=',energy)
    return energy

def softplus(x):
    return np.log(1+np.exp(x))

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def hidden_layer(x,u,v,visible_bias,hidden_bias):
    num=x.shape[0]
    x=np.reshape(x,(num,75,50))
    hidden_layer=np.zeros((num,40,40),dtype=np.float32)
    for i in range(num):
        x_image=x[i]
        hidden_layer[i] = sigmoid(np.matmul(np.matmul(u, x_image), np.transpose(v)) + hidden_bias)  
    return hidden_layer


def excel1m(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    c1 = np.arange(0, nrows, 1)
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        minVals = min(cols)
        maxVals = max(cols)
        cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
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
    c1 = np.arange(0, nrows, 1)
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        minVals = min(cols)
        maxVals = max(cols)
        cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        #ranges = maxVals - minVals
        #b = cols1 - minVals
        #normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = cols # 把数据进行存储
        

    return datamatrix


data = excel1m('1test_v.xlsx')
#data_test = excel2m('D:/coding/myself/vedio_test.xlsx')

w_u=excel2m('video_u.xlsx')   ##30*75
w_v=excel2m('video_v.xlsx')    ##30*50
visible_bias=excel2m('video_vbias.xlsx')   ##75*50
hidden_bias=excel2m('video_hbias.xlsx')    ##30*30

data=np.reshape(data,(2590,75,50))
#data_test=np.reshape(data_test,(2000,75,50))



'''
matrix_3=hidden_layer(data,w_u,w_v,visible_bias,hidden_bias)
print(np.shape(matrix_3))
matrix_2=np.reshape(matrix_3,(2590,1600))
hidden=pd.DataFrame(matrix_2)
writer1=pd.ExcelWriter('1test_hidden_v.xlsx')
hidden.to_excel(writer1,'Sheet1')
writer1.save()
'''

'''
energy=[]
for i in range(3000):
  aa=data[i:(i+1),:,:]
  e=free_energy(aa,w_u,w_v,visible_bias,hidden_bias)
  energy.append(e)
#print(energy)  

x1=np.arange(0,1029,1)
y1=energy
plot=plt.plot(x1,y1,linewidth=2)

plt.axvline(x=187,ls=':',c='red')
plt.axvline(x=595,ls=':',c='red')
plt.axvline(x=840,ls=':',c='red')
plt.axvline(x=962,ls=':',c='red')
#plt.axvline(x=170,ls=':',c='red')
#plt.axvline(x=200,ls=':',c='red')
#plt.axvline(x=600,ls=':',c='red')
#plt.axvline(x=600,ls=':',c='red')
#plt.axvline(x=600,ls=':',c='red')

plt.xlabel('minibatches')
plt.ylabel('free_energy')
#plt.grid()
plt.show()
'''
'''
#energy=free_energy(data,w_u,w_v,visible_bias,hidden_bias)
#print(energy)
n1=0
delta_n=20
data1=data[0:20,:,:]
energy1=free_energy(data1,w_u,w_v,visible_bias,hidden_bias)
#print(energy1)


for i in range(40):
  n2=n1+delta_n
  data2=data[n2:(n2+delta_n),:]
  energy2=free_energy(data2,w_u,w_v,visible_bias,hidden_bias)
  print(energy2)
  delta_energy=np.abs((energy2-energy1)/200)
  print('delta_energy',delta_energy)
  if delta_energy>0.08:
    delta_n=20
  else:
    delta_n=min(60,2*delta_n)

  energy1=energy2
  n1=n2
  print('delta_n',delta_n)
'''






n1=0
delta_n=20

a1=data[0:20,:,:]
delta_e=[]
window_size=[]
for i in range(delta_n):
  delta_e.append(0)
  window_size.append(20)
energy1=free_energy(a1,w_u,w_v,visible_bias,hidden_bias)

print(energy1)


for i in range(1000):
  n2=n1+delta_n
  b1=data[n2:(n2+delta_n),:,:]
  print('b1',type(b1))
  
  if np.sum(b1)==0:
    break
  
 
  energy2=free_energy(b1,w_u,w_v,visible_bias,hidden_bias)

  print(energy2)

  delta_energy=np.abs((energy2-energy1)/-30)
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
  aa=data[i:(i+1),:,:]
  e=free_energy(aa,w_u,w_v,visible_bias,hidden_bias)

  energy.append(e)

figsize=13,8
figure, ax=plt.subplots(figsize=figsize)
x1=np.arange(0,2590,1)
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
x1=np.arange(0,len(delta_e),1)
#plot=plt.plot(x1,y1,linewidth=4,c='blue')
#plot=plt.plot(x1,y2,linewidth=4,c='blue',label='video input')
plot=plt.plot(x1,delta_e,linewidth=4)
#plot=plt.plot(x1,y4,linewidth=3,c='c',label='m3')
plt.ylim(0,1)
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
x1=np.arange(0,len(window_size),1)
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










           
