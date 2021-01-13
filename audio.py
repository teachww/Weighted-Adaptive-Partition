
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

class RBM(object):
  """ Restricted Boltzmann Machine (RBM) """
  def __init__(self, input=None, n_visible=534, n_hidden=300, \
      W=None, hbias=None, vbias=None, numpy_rng=None):

    self.input = input
    self.n_visible = n_visible
    self.n_hidden = n_hidden

    if numpy_rng is None:
      numpy_rng = numpy.random.RandomState(1234)

    if W is None:

      W = numpy.asarray(numpy_rng.uniform(
        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
        size=(n_visible, n_hidden)))

      #W=numpy.asarray(numpy_rng.uniform(-0.01,0.01,size=(n_visible,n_hidden)))

    if hbias is None:
      hbias = numpy.zeros(n_hidden)

    if vbias is None:
      vbias = numpy.zeros(n_visible)

    self.numpy_rng = numpy_rng
    self.W = W
    self.hbias = hbias
    self.vbias = vbias


  ##定义基于输入层的能量函数值
  def free_energy(self, x):
    n=x.shape[0]
    free_energy=0
    for i in range(n):
      #print('###########################################################',i,'#############################################################')
      v_sample=x[i]
      #free_energy=-numpy.trace(numpy.dot(numpy.transpose(self.vbias),v_sample))-numpy.sum(self.softplus(numpy.dot(numpy.transpose(v_sample),self.W)+self.hbias))
      free_energy+=(-numpy.dot(self.vbias,numpy.transpose(v_sample))-numpy.sum(self.softplus(numpy.dot(v_sample,self.W)+self.hbias)))/n
      #print('e1=',numpy.dot(self.vbias,numpy.transpose(v_sample)))
      #print('e2=',numpy.dot(v_sample,self.W))
      #print('e3=',self.softplus(numpy.dot(v_sample,self.W)+self.hbias))
    return free_energy


    
  def softplus(self,x):
      return numpy.log(1+numpy.exp(x))

  def sigmoid(self,x):
    return 1/(1+numpy.exp(-x))

  ##输入层==>隐含层的概率
  def propup(self, vis):
    pre_sigmoid_activation = numpy.dot(vis, self.W) + self.hbias
    return self.sigmoid(pre_sigmoid_activation)

  ##得到hidden的二进制状态
  def sample_h_given_v(self, v0_sample):
    h1_mean = self.propup(v0_sample)
    #h1_sample = self.numpy_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
    h1_sample = h1_mean
    return [h1_mean, h1_sample]

  ##隐含层==>输入层的概率
  def propdown(self, hid):
    pre_sigmoid_activation = numpy.dot(hid, self.W.T) + self.vbias
    return self.sigmoid(pre_sigmoid_activation)

  ##重构输入层的二进制状态
  def sample_v_given_h(self, h0_sample):
    v1_mean = self.propdown(h0_sample)
    #v1_sample = self.numpy_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
    v1_sample = v1_mean
    return [v1_mean, v1_sample]

  def gibbs_hvh(self, h0_sample):
    v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
    h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
    return [v1_mean, v1_sample,
            h1_mean, h1_sample]

  def gibbs_vhv(self, v0_sample):
    """ no use """
    return
    

    

  ##进行更新
  def get_cost_updates(self,x,lr, persistent=None, k=1):
    ph_mean, ph_sample = self.sample_h_given_v(self.input)

    if persistent is None:
      chain_start = ph_sample
    else:
      chain_start = persistent

    for step in range(k):
      if step == 0:
        nv_means, nv_samples,\
        nh_means, nh_samples = self.gibbs_hvh(chain_start)
      else:
        nv_means, nv_samples,\
        nh_means, nh_samples = self.gibbs_hvh(nh_samples)

    #print('hidden_p=',ph_mean)
    #print('hidden_recon_p=',nh_means)
    self.W += lr * (numpy.dot(self.input.T, ph_mean) - numpy.dot(nv_samples.T, nh_means))
    self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
    self.hbias += lr * numpy.mean(ph_mean - nh_means, axis=0)
    #print('weights=',self.W)
    #print('visible_bias=',self.vbias)
    #print('hidden_bias=',self.hbias)
    #print('visible_recon_p',nv_samples)
    #energy=self.free_energy(x)
    return self.W,self.vbias,self.hbias

  def get_reconstruction_cross_entropy(self):
      pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
      sigmoid_activation_h = self.sigmoid(pre_sigmoid_activation_h)
      #print('pre_sigmoid_activation_h=',sigmoid_activation_h)

      pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
      sigmoid_activation_v = self.sigmoid(pre_sigmoid_activation_v)
      #print('pre_sigmoid_activation_v=', sigmoid_activation_v)
      #print('input=', self.input)

      cross_entropy = - numpy.mean(
          numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
                    (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                    axis=1))

      return cross_entropy

  def get_hidden(self):
      hidden=self.sigmoid(numpy.dot(self.input, self.W) + self.hbias)
      return hidden




  def get_pseudo_likelihood_cost(self, updates):
    """ no use """
    return


def save(data,path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h,l] = data.shape  # h为行数，l为列数
    for i in range(h): 
        for j in range(l):
            sheet1.write(i,j,data[i,j])
    f.save(path)


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
        ranges = maxVals - minVals
        b = cols1 - minVals
        normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = normcols # 把数据进行存储
    return datamatrix
  

data = excel2m('train_a.xlsx')
print(data)
#save(data,'train.xlsx')
#print(data)
rbm=RBM()

####################training model######################
rbm=RBM(input=data)
for epoch in range(3000):
  rbm.get_cost_updates(data,lr=0.001,k=1)
  cost=rbm.get_reconstruction_cross_entropy()
  print('Training epoch %d,cost is %d'%(epoch,cost))

  #print('#############################################################',epoch,'#############################################################')
#########################################################
hidden=rbm.get_hidden()
print(hidden)


'''
########################energy##########################
energy=[]
for i in range(1029):
  e=rbm.free_energy(data[i])
  energy.append(e)

x1=numpy.arange(0,1029,1)
y1=energy
plot=plt.plot(x1,y1,linewidth=2)
#plt.ylim(0,-10000000)
plt.xlabel('minibatches')
plt.ylabel('free_energy')
plt.show()

#energy=pd.DataFrame(energy)
#writer1=pd.ExcelWriter('a_energy.xlsx')
#energy.to_excel(writer1,'Sheet1')
#writer1.save()


########################################################

'''  



weights=pd.DataFrame(rbm.W)
hidden_bias=pd.DataFrame(rbm.hbias) 
visible_bias=pd.DataFrame(rbm.vbias)
writer1=pd.ExcelWriter("audio_weights.xlsx")
writer2=pd.ExcelWriter('audio_hbias.xlsx')
writer3=pd.ExcelWriter('audio_vbias.xlsx')
weights.to_excel(writer1,'Sheet1')
hidden_bias.to_excel(writer2,'Sheet1')
visible_bias.to_excel(writer3,'Sheet1')
writer1.save()
writer2.save()
writer3.save()


'''

n1=0
delta_n=20
data1=data[0:20,:]
energy1=rbm.free_energy(data1)

for i in range(40):
  n2=n1+delta_n
  data2=data[n2:(n2+delta_n),:]
  energy2=rbm.free_energy(data2)
  delta_energy=numpy.abs((energy2-energy1)/energy)
  print('delta_energy',delta_energy)
  if delta_energy>0.08:
    delta_n=20
  else:
    delta_n=min(60,2*delta_n)

  energy1=energy2
  n1=n2
  print('delta_n',delta_n)


'''


































