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

class RBM(object):
    def __init__(self):

        
        ##weights and biases
        self.u = np.random.uniform(-.01,0.01,(40,75))
        self.v = np.random.uniform(-.01,0.01,(40,50))
        self.visible_bias = np.zeros((75,50))
        self.hidden_bias = np.zeros((40,40))

        self.delta_u = np.zeros((40,75))
        self.delta_v = np.zeros((40,50))
        self.delta_visible_bias = np.zeros((75,50))
        self.delta_hidden_bias = np.zeros((40,40))

        self.update_delta_u = np.zeros((40,75))
        self.update_delta_v = np.zeros((40,50))
        self.update_delta_visible_bias = np.zeros((75,50))
        self.update_delta_hidden_bias = np.zeros((40,40))

    def fit(self,x,max_epoch=1,learning_rate=0.001,T=2000,w_b=0.01,momentum=0.5):
        batch_size = x.shape[0]
        #num_batch = num // 100

        for t in range(T):

            delta_u =np.zeros((40,75))
            delta_v =np.zeros((40,50))
            delta_visible_bias =  np.zeros((75,50))
            delta_hidden_bias =np.zeros((40,40))
            for j in range(batch_size):
                
                x_image=x[j]

                for k in range(max_epoch):
                    xu_1 = np.matmul(self.u, x_image)
                    hidden_p = self.sigmoid(np.matmul(xu_1, np.transpose(self.v)) + self.hidden_bias)  ##[15,15]
                    #y_1 = self.sample_bernoulli(hidden_p)
                    y_1 = hidden_p

                    uy = np.matmul(np.transpose(self.u), y_1)  ##[28,15]
                    visible_recon_p = self.sigmoid(np.matmul(uy, self.v) + self.visible_bias)  ##[28,28]
                    #v_recon = self.sample_bernoulli(visible_recon_p)
                    v_recon = visible_recon_p

                    hidden_p_recon = self.sigmoid(np.matmul(np.matmul(self.u, v_recon), np.transpose(self.v)) + self.hidden_bias)  ##[15,15]

                delta_u += -learning_rate*(np.matmul(np.matmul(hidden_p_recon, self.v), np.transpose(v_recon)) - np.matmul(np.matmul(hidden_p, self.v), np.transpose(x_image)))/batch_size
                delta_v += -learning_rate*(np.matmul(np.matmul(np.transpose(hidden_p_recon), self.u), v_recon) - np.matmul(np.matmul(np.transpose(hidden_p), self.u), x_image))/batch_size
                delta_visible_bias += -learning_rate* (v_recon - x_image)/batch_size
                delta_hidden_bias +=  -learning_rate*(hidden_p_recon - hidden_p)/batch_size

            self.update_delta_u = momentum*self.update_delta_u+delta_u-learning_rate*w_b*self.u
            self.update_delta_v = momentum*self.update_delta_v+delta_v-learning_rate*w_b*self.v
            self.update_delta_visible_bias =momentum*self.update_delta_visible_bias+delta_visible_bias
            self.update_delta_hidden_bias = momentum*self.update_delta_hidden_bias+delta_hidden_bias

            self.u=self.u+self.update_delta_u
            self.v=self.v+self.update_delta_v
            self.visible_bias=self.visible_bias+self.update_delta_visible_bias
            self.hidden_bias=self.hidden_bias+self.update_delta_hidden_bias


            u=self.free_energy(x)
            print('t=',t,'energy=',u)
        #print('u=',self.u,'v=',self.v,'visible_u=',self.visible_bias,'visible_v=',self.hidden_bias)

        return self.u,self.v,self.visible_bias,self.hidden_bias


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def mse(self,x):
        mse=np.zeros([1])
        batch_size=x.shape[0]
        for r in range(batch_size):
            x_image = x[r]
            h_p = self.sigmoid(np.matmul(np.matmul(self.u, x_image), np.transpose(self.v)) + self.hidden_bias)  ##[15,15]
            y = self.sample_bernoulli(h_p)

            v_recon_p = self.sigmoid(np.matmul(np.matmul(np.transpose(self.u), y), self.v) + self.visible_bias)  ##[28,28]
            x_recon = self.sample_bernoulli(v_recon_p)
            mse = (mse + np.sum(x_recon - x_image)) / 100
        return mse

    def relu(self,x):
        s=np.where(x<0,0,x)
        return s

    def sample_bernoulli(self,probs):
        return self.relu(np.sign(probs - np.random.uniform(size=np.shape(probs))))


    def reconstruct(self,X):
        recon=np.zeros(shape=np.shape(X))
        n=X.shape[0]
        for i in range(n):
            x_image=X[i]
            '''
            data_1 = x_image * 255
            new_im_1 = Image.fromarray(data_1.astype(np.uint8))
            plt.imshow(data_1, cmap=plt.cm.gray, vmin=0, vmax=255)
            new_im_1.show()
            '''
            h_p=self.sigmoid(np.matmul(np.matmul(self.u,x_image),np.transpose(self.v))+self.hidden_bias) ##[15,15]
            #y_1=sample_bernoulli(hidden_p)
            ##可见层的重构
            v_recon_p=self.sigmoid(np.matmul(np.matmul(np.transpose(self.u),h_p),self.v)+self.visible_bias) ##[28,28]
            v_recon=self.sample_bernoulli(v_recon_p)
            recon[i]=v_recon
            '''
            data=v_recon*255
            new_im=Image.fromarray(data.astype(np.uint8))
            plt.imshow(data,cmap=plt.cm.gray,vmin=0,vmax=255)
            new_im.show()
            '''
            #print(v_recon)


        return recon

    def free_energy(self,x):
        num=x.shape[0]
        energy=np.zeros([1])
        for i in range(num):
            x_image=x[i]
            energy=energy- np.trace(np.matmul(np.transpose(x_image), self.visible_bias)) - np.sum(self.softplus(np.trace(np.matmul(np.matmul(self.u, x_image), np.transpose(self.v)) + self.hidden_bias)))
        energy=energy/num
        #print('energy=',energy)
        return energy

    def softplus(self,x):
        return np.log(1+np.exp(x))

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
        ranges = maxVals - minVals
        b = cols1 - minVals
        normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = normcols # 把数据进行存储
    return datamatrix


data = excel2m('D:/paper/myself/cuave/g11/new/train_v.xlsx')
#data_test = excel2m('D:/coding/mypaper/vedio_test.xlsx')

data=np.reshape(data,(3000,75,50))
#data_test=np.reshape(data_test,(2000,75,50))
rbm=RBM()

########################training model##########################
#data=np.reshape(data,(1000,75,50))
rbm.fit(data)
################################################################

'''

energy=[]
for i in range(1029):
  aa=data[i:(i+1),:,:]
  e=rbm.free_energy(aa)
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
W_u=pd.DataFrame(rbm.u)
W_v=pd.DataFrame(rbm.v)
hidden_bias=pd.DataFrame(rbm.hidden_bias) 
visible_bias=pd.DataFrame(rbm.visible_bias)
writer1=pd.ExcelWriter("video_u.xlsx")
writer2=pd.ExcelWriter('video_v.xlsx')
writer3=pd.ExcelWriter('video_hbias.xlsx')
writer4=pd.ExcelWriter('video_vbias.xlsx')
W_u.to_excel(writer1,'Sheet1')
W_v.to_excel(writer2,'Sheet1')
hidden_bias.to_excel(writer3,'Sheet1')
visible_bias.to_excel(writer4,'Sheet1')
writer1.save()
writer2.save()
writer3.save()
writer4.save()



'''
##############################sliding window##########################
#energy=rbm.free_energy(data[50:100])
#print(energy)


n1=0
delta_n=50
data1=data_test[0:50,:,:]
#energy=rbm.free_energy(data)
energy1=rbm.free_energy(data1)

for i in range(50):
  n2=n1+delta_n
  data2=data_test[n2:(n2+delta_n),:]
  energy2=rbm.free_energy(data2)
  delta_energy=np.abs((energy2-energy1)/energy1)
  print('delta_energy',delta_energy)
  if delta_energy>0.043:
    delta_n=50
  else:
    delta_n=min(120,2*delta_n)

  energy1=energy2
  n1=n2
  print('delta_n',delta_n)


######################################################################

'''

