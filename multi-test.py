import pandas as pd
import numpy 
import scipy
import csv
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import xlwt
import xlrd


def free_energy1(x,W,vbias,hbias):
  n=x.shape[0]
  free_energy=0
  for i in range(n):
    v_sample=x[i]
    #free_energy+=(-numpy.dot(v_sample,vbias)-numpy.sum(softplus(numpy.dot(v_sample,W)+hbias)))/n
    #free_energy += (-numpy.dot(vbias, numpy.transpose(v_sample)) - numpy.sum(softplus(numpy.dot(v_sample,W) + hbias))) / n
    free_energy+=(-numpy.dot(v_sample,vbias)-numpy.sum(softplus(numpy.dot(v_sample,W)+numpy.transpose(hbias))))
  free_energy=free_energy/n
    #print(vbias)
  return free_energy

def free_energy2(x,w_u,w_v,visible_bias,hidden_bias):
    num=x.shape[0]
    free_energy=0
    for i in range(num):
        x_image=x[i]   ##75*50
        free_energy+= -numpy.sum(numpy.matmul(numpy.transpose(x_image),visible_bias)) - numpy.sum(softplus(numpy.matmul(numpy.matmul(w_u, x_image), numpy.transpose(w_v)) + hidden_bias))
        #free_energy+= -numpy.trace(numpy.matmul(numpy.transpose(x_image),visible_bias)) - numpy.sum(softplus(numpy.trace(numpy.matmul(numpy.matmul(w_u, x_image), numpy.transpose(w_v)) + hidden_bias)))
       
    free_energy=free_energy/num
    #print('energy=',energy)
    return free_energy

def free_energy3(x,y,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias):
  num=x.shape[0]
  free_energy=numpy.zeros([1])
  for i in range(num):
      x_image=x[i]
      y_image=y[i]
      #free_energy+=- numpy.trace(numpy.matmul(numpy.transpose(x_image), h1_bias)) - numpy.trace(numpy.matmul(numpy.transpose(y_image), h2_bias))- numpy.sum(softplus(numpy.trace(numpy.matmul(numpy.matmul(u1, x_image), numpy.transpose(v1))+ numpy.matmul(numpy.matmul(u2, y_image), numpy.transpose(v2)) + h3_bias)))
      free_energy+= -numpy.sum(numpy.matmul(numpy.transpose(x_image), h1_bias)) -numpy.sum(numpy.matmul(numpy.transpose(y_image), h2_bias))- numpy.sum(softplus(numpy.matmul(numpy.matmul(u1, x_image), numpy.transpose(v1))+ numpy.matmul(numpy.matmul(u2, y_image), numpy.transpose(v2)) + h3_bias))
  free_energy=free_energy/num


  return free_energy

def softplus(x):
  return numpy.log(1+numpy.exp(x))
  
def sigmoid(x):
    #return numpy.exp(x)/(1+numpy.exp(x))
    return 1/(1+numpy.exp(-x))
  
def weights1(data):
    m=numpy.shape(data)[0]
    delta_pro=0
    data.sort()
    for i in range(m):

        pro=data[i][1]-data[i][0]

        pro=pro/(numpy.abs(numpy.log(data[i][0]+1e-320))+numpy.abs(numpy.log(data[i][1]+1e-320)))
        delta_pro+=pro
        #print(numpy.log(data[i][0]),numpy.log(data[i][1]))
    delta_pro=delta_pro/m
    return delta_pro

def weights(data):
    m=numpy.shape(data)[0]
    delta_pro=0
    data.sort()
    #print('data=',data)
    for i in range(m):

        pro=data[i][1]-data[i][0]

        pro=pro/(numpy.abs(numpy.log(data[i][0]+1e-320))+numpy.abs(numpy.log(data[i][1]+1e-320))+numpy.abs(numpy.log(data[i][2]+1e-320))+numpy.abs(numpy.log(data[i][3]+1e-320)))
        delta_pro+=pro
        #print(numpy.log(data[i][0]),numpy.log(data[i][1]))
    delta_pro=delta_pro/m
    return delta_pro


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



audio_data = excel1m('1test_a.xlsx')

audio_weights=excel2m('3vector_W.xlsx')
audio_hidden_bias=excel2m('3vector_hb.xlsx')
audio_visible_bias=excel2m('3vector_vb.xlsx')

video_data = excel1m('1test_v.xlsx')
video_data=numpy.reshape(video_data,(2590,75,50))

video_w_u=excel2m('video_u.xlsx')   ##30*75
video_w_v=excel2m('video_v.xlsx')    ##30*50
video_visible_bias=excel2m('video_vbias.xlsx')   ##75*50
video_hidden_bias=excel2m('video_hbias.xlsx')    ##30*30


data_audio = excel2m('1test_hidden_a.xlsx')
data_video = excel2m('1test_hidden_v.xlsx')
u1=excel2m('multi_u11.xlsx')
v1=excel2m('multi_v11.xlsx')
u2=excel2m('multi_u21.xlsx')
v2=excel2m('multi_v21.xlsx')
h1_bias=excel2m('multi_h1_bias1.xlsx')
h2_bias=excel2m('multi_h2_bias1.xlsx')
h3_bias=excel2m('multi_h3_bias1.xlsx')
audio_pro=excel2m('3audio_dis.xlsx')
video_pro=excel2m('3video_dis.xlsx')

data1=data_audio
data2=data_video

a=numpy.zeros((2590,100),dtype=numpy.float32)
print(numpy.shape(data1))



c=numpy.column_stack((data1,a))
h_audio=numpy.reshape(c,(2590,20,20))
#print('h_audio=',h_audio)

#######################video_hidden########################
h_video=numpy.reshape(data2,(2590,40,40))
#print('h_video=',h_video)



energy1=[]
energy2=[]
energy3=[]

for i in range(2590):
  a1=audio_data[i:(i+1)]
  a2=video_data[i:(i+1),:,:]
  
  a3 = h_audio[i:(i + 1)]
  a4 = h_video[i:(i + 1)]

  '''
  w1=weights1(audio_pro[i:(i+1)])
  w2=weights1(video_pro[i:(i+1)])

  a3 =(w1/(w1+w2))* h_audio[i:(i + 1)]
  a4 = (w2/(w1+w2))*h_video[i:(i + 1)]
  '''
  e1=free_energy1(a1,audio_weights,audio_visible_bias,audio_hidden_bias)

  
  e2=free_energy2(a2,video_w_u,video_w_v,video_visible_bias,video_hidden_bias)

  e3=free_energy3(a3,a4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
  #print(e3)
  e=e1+e2+e3
  #energy1.append(e1)
  #energy2.append(e2)
  energy3.append(e)





x1=numpy.arange(0,2590,1)
#y1=energy1
#y2=energy2
y3=energy3


figsize=13,8
figure, ax=plt.subplots(figsize=figsize)

#plot=plt.plot(x1,y1,linewidth=4,c='blue')
#plot=plt.plot(x1,y2,linewidth=4,c='blue',label='video input')
plot=plt.plot(x1,y3,linewidth=4,c='orange',label='multi-modal input')
#plot=plt.plot(x1,y4,linewidth=3,c='c',label='m3')
#plt.ylim(0,-10000000)
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



def free_energy(energy,n1,n2):
    e=energy[n1:n2]
    num=(n2-n1)
    energy=sum(e)/num
    return energy


'''
energy3=pd.DataFrame(energy3)
writer=pd.ExcelWriter('energy31.xlsx')
energy3.to_excel(writer,'Sheet1')
writer.save()


n1=0
delta_n=20
a1=audio_data[0:20,:]
a2=video_data[0:20,:]


#a3 = h_audio[0:20]
#a4 = h_video[0:20]



w1 = weights1(audio_pro[0:20,:])
w2 = weights1(video_pro[0:20,:])
print('w1=',w1 / (w1 + w2))
print('w2=',w2 / (w1 + w2))
a3 = (w1 / (w1 + w2)) * h_audio[0:20]
a4 = (w2 / (w1 + w2)) * h_video[0:20]

#energy1=free_energy3(a3,a4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
#energy1=free_energy2(a2,video_w_u,video_w_v,video_visible_bias,video_hidden_bias)
#energy1=free_energy3(a3,a4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
energy1=free_energy1(a1,audio_weights,audio_visible_bias,audio_hidden_bias)+free_energy2(a2,video_w_u,video_w_v,video_visible_bias,video_hidden_bias)+free_energy3(a3,a4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
#energy1=free_energy(energy3,n1,n1+delta_n)
print(energy1)

for i in range(1000):
  n2=n1+delta_n
  b1=audio_data[n2:(n2+delta_n),:]
  b2=video_data[n2:(n2+delta_n),:]

  #b3 = h_audio[n2:(n2 + delta_n), :, :]
  #b4 = h_video[n2:(n2 + delta_n), :, :]

  
  w3 = weights1(audio_pro[n2:(n2+delta_n),:])
  w4 = weights1(video_pro[n2:(n2+delta_n),:])
  print('w1=',w3 / (w3+w4))
  print('w2=',w4 / (w3+w4))
  b3=(w3/(w3+w4))*h_audio[n2:(n2+delta_n),:,:]
  b4=(w4/(w3+w4))*h_video[n2:(n2+delta_n),:,:]
  

  #energy2=free_energy1(b1,audio_weights,audio_visible_bias,audio_hidden_bias)
  #energy2=free_energy2(b2,video_w_u,video_w_v,video_visible_bias,video_hidden_bias)
  #energy2=free_energy3(b3,b4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
  energy2=free_energy1(b1,audio_weights,audio_visible_bias,audio_hidden_bias)+free_energy2(b2,video_w_u,video_w_v,video_visible_bias,video_hidden_bias)+free_energy3(b3,b4,u1,v1,u2,v2,h1_bias,h2_bias,h3_bias)
  #energy2=free_energy(energy3,n2,n2+delta_n)

  print(energy2)

  delta_energy=numpy.abs((energy2-energy1)/-1400)
  print('energy_change,',(energy2-energy1),'delta_energy',delta_energy)
  if delta_energy>0.08:
    delta_n=20
  else:
    delta_n=min(60,2*delta_n)

  energy1=energy2
  n1=n2
  print('delta_n',delta_n)




'''





























