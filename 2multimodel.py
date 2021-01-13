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
        self.u1 = np.random.uniform(-.01,0.01,(25,20))
        self.v1 = np.random.uniform(-.01,0.01,(25,20))
        self.u2 = np.random.uniform(-.01,0.01,(25,40))
        self.v2 = np.random.uniform(-.01,0.01,(25,40))
        self.h1_bias = np.zeros((20,20))
        self.h2_bias = np.zeros((40,40))
        self.h3_bias = np.zeros((25,25))

        self.delta_u1 = np.zeros((25,20))
        self.delta_v1 = np.zeros((25,20))
        self.delta_u2 = np.zeros((25,40))
        self.delta_v2 = np.zeros((25,40))
        self.delta_h1_bias = np.zeros((20,20))
        self.delta_h2_bias = np.zeros((40,40))
        self.delta_h3_bias = np.zeros((25,25))

        self.update_u1 = np.zeros((25,20))
        self.update_v1 = np.zeros((25,20))
        self.update_u2 = np.zeros((25,40))
        self.update_v2 = np.zeros((25,40))
        self.update_h1_bias = np.zeros((20,20))
        self.update_h2_bias = np.zeros((40,40))
        self.update_h3_bias = np.zeros((25,25))
        

    def fit(self,x,y,max_epoch=1,learning_rate=0.001,T=2000,w_b=0.01,momentum=0.5):
        batch_size = x.shape[0]
        #num_batch = num // 100

        for t in range(T):

            delta_u1 =np.zeros((25,20))
            delta_v1 =np.zeros((25,20))
            delta_u2 =np.zeros((25,40))
            delta_v2 =np.zeros((25,40))
            delta_h1_bias =np.zeros((20,20))
            delta_h2_bias =np.zeros((40,40))
            delta_h3_bias =np.zeros((25,25))
            for j in range(batch_size):
                
                x_image=x[j]
                y_image=y[j]
                #b=np.zeros((260,40),dtype=np.float32)
                #c=np.zeros((300,260),dtype=np.float32)
                #d=np.row_stack((hidden_y,b))
                #y_image=np.column_stack((d,c))
                #print('x_image=',x_image)
                #print('y_image=',y_image)

                for k in range(max_epoch):
                    hidden_p = self.sigmoid(np.matmul(np.matmul(self.u1, x_image), np.transpose(self.v1))+np.matmul(np.matmul(self.u2, y_image), np.transpose(self.v2)) + self.h3_bias)
                    #y_1 = self.sample_bernoulli(hidden_p)
                    y_1 = hidden_p   ##(200,200)

                    uy1 = np.matmul(np.transpose(self.u1), y_1)  ##(300,200)
                    visible_recon_p1 = self.sigmoid(np.matmul(uy1, self.v1) + self.h1_bias)  
                    #v_recon = self.sample_bernoulli(visible_recon_p)
                    v_recon1 = visible_recon_p1

                    uy2 = np.matmul(np.transpose(self.u2), y_1)  
                    visible_recon_p2 = self.sigmoid(np.matmul(uy2, self.v2) + self.h2_bias)  
                    #v_recon = self.sample_bernoulli(visible_recon_p)
                    v_recon2 = visible_recon_p2

                    #hidden_p_recon = self.sigmoid(np.matmul(np.matmul(self.u1, v_recon), np.transpose(self.v1)) + self.h3_bias)
                    hidden_p_recon = self.sigmoid(np.matmul(np.matmul(self.u1, v_recon1), np.transpose(self.v1))+np.matmul(np.matmul(self.u2, v_recon2), np.transpose(self.v2)) + self.h3_bias)

                delta_u1 += -learning_rate*(np.matmul(np.matmul(hidden_p_recon, self.v1), np.transpose(v_recon1)) - np.matmul(np.matmul(hidden_p, self.v1), np.transpose(x_image)))/batch_size
                delta_v1 += -learning_rate*(np.matmul(np.matmul(np.transpose(hidden_p_recon), self.u1), v_recon1) - np.matmul(np.matmul(np.transpose(hidden_p), self.u1), x_image))/batch_size
                delta_u2 += -learning_rate*(np.matmul(np.matmul(hidden_p_recon, self.v2), np.transpose(v_recon2)) - np.matmul(np.matmul(hidden_p, self.v2), np.transpose(y_image)))/batch_size
                delta_v2 += -learning_rate*(np.matmul(np.matmul(np.transpose(hidden_p_recon), self.u2), v_recon2) - np.matmul(np.matmul(np.transpose(hidden_p), self.u2), y_image))/batch_size

                delta_h1_bias += -learning_rate* (v_recon1 - x_image)/batch_size
                delta_h2_bias += -learning_rate* (v_recon2 - y_image)/batch_size
                delta_h3_bias +=  -learning_rate*(hidden_p_recon - hidden_p)/batch_size

            self.update_u1 = momentum*self.update_u1+delta_u1-learning_rate*w_b*self.u1
            self.update_v1 = momentum*self.update_v1+delta_v1-learning_rate*w_b*self.v1
            self.update_u2 = momentum*self.update_u2+delta_u2-learning_rate*w_b*self.u2
            self.update_v2 = momentum*self.update_v2+delta_v2-learning_rate*w_b*self.v2
            self.update_h1_bias =momentum*self.update_h1_bias+delta_h1_bias
            self.update_h2_bias =momentum*self.update_h2_bias+delta_h2_bias
            self.update_h3_bias =momentum*self.update_h3_bias+delta_h3_bias

            self.u1=self.u1+self.update_u1
            self.v1=self.v1+self.update_v1
            self.u2=self.u2+self.update_u2
            self.v2=self.v2+self.update_v2
            self.h1_bias=self.h1_bias+self.update_h1_bias
            self.h2_bias=self.h2_bias+self.update_h2_bias
            self.h3_bias=self.h3_bias+self.update_h3_bias


            #u=self.mse(x)
            print('t=',t,)
        #print('u=',self.u,'v=',self.v,'visible_u=',self.visible_bias,'visible_v=',self.hidden_bias)

        return self.u1,self.v1,self.u2,self.v2,self.h1_bias,self.h2_bias,self.h3_bias


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

    def free_energy(self,x,y):
        num=x.shape[0]
        energy=np.zeros([1])
        for i in range(num):
            x_image=x[i]
            y_image=y[i]
            energy=energy- np.trace(np.matmul(np.transpose(x_image), self.h1_bias)) -np.trace(np.matmul(np.transpose(y_image), self.h2_bias))-  np.sum(self.softplus(np.trace(np.matmul(np.matmul(self.u1, x_image), np.transpose(self.v1)) + np.matmul(np.matmul(self.u2, y_image), np.transpose(self.v2))+ self.h3_bias)))
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
        #ranges = maxVals - minVals
        #b = cols1 - minVals
        #normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = cols # 把数据进行存储
    return datamatrix


def weights(data):
    m=np.shape(data)[0]
    delta_pro=0
    data.sort()
    for i in range(m):

        pro=data[i][1]-data[i][0]

        #pro=pro/(numpy.abs(numpy.log(data[i][0]+1e-320))+numpy.abs(numpy.log(data[i][1]+1e-320))+numpy.abs(numpy.log(data[i][2]+1e-320))+numpy.abs(numpy.log(data[i][3]+1e-320)))
        delta_pro+=pro
        #print(numpy.log(data[i][0]),numpy.log(data[i][1]))
    delta_pro=delta_pro/m
    return delta_pro


#audio_pro=excel2m('D:/paper/myself/cuave/g11/audio_dis.xlsx')
#video_pro=excel2m('D:/paper/myself/cuave/g11/video_dis.xlsx')
data_audio = excel2m('D:/paper/myself/cuave/g11/new/hidden_a.xlsx')
data_video = excel2m('D:/paper/myself/cuave/g11/new/hidden_v.xlsx')


data1=data_audio
data2=data_video
#w1=weights(data1)
#w2=weights(data2)
#data1=w1/(w1+w2)*data1
#data2=w2/(w1+w2)*data2



a=np.zeros((3000,100),dtype=np.float32)
c=np.column_stack((data1,a))
h_audio=np.reshape(c,(3000,20,20))

#######################video_hidden########################
h_video=np.reshape(data2,(3000,40,40))


###########################################################



rbm=RBM()

########################training model##########################
#data=np.reshape(data,(1000,75,50))
rbm.fit(h_audio,h_video)
################################################################






W_u1=pd.DataFrame(rbm.u1)
W_v1=pd.DataFrame(rbm.v1)
W_u2=pd.DataFrame(rbm.u2)
W_v2=pd.DataFrame(rbm.v2)
h1_bias=pd.DataFrame(rbm.h1_bias) 
h2_bias=pd.DataFrame(rbm.h2_bias)
h3_bias=pd.DataFrame(rbm.h3_bias)
writer1=pd.ExcelWriter("multi_u11.xlsx")
writer2=pd.ExcelWriter('multi_v11.xlsx')
writer3=pd.ExcelWriter('multi_u21.xlsx')
writer4=pd.ExcelWriter('multi_v21.xlsx')
writer5=pd.ExcelWriter('multi_h1_bias1.xlsx')
writer6=pd.ExcelWriter('multi_h2_bias1.xlsx')
writer7=pd.ExcelWriter('multi_h3_bias1.xlsx')

W_u1.to_excel(writer1,'Sheet1')
W_v1.to_excel(writer2,'Sheet1')
W_u2.to_excel(writer3,'Sheet1')
W_v2.to_excel(writer4,'Sheet1')
h1_bias.to_excel(writer5,'Sheet1')
h2_bias.to_excel(writer6,'Sheet1')
h3_bias.to_excel(writer7,'Sheet1')

writer1.save()
writer2.save()
writer3.save()
writer4.save()

writer5.save()
writer6.save()
writer7.save()




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

