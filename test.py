import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.linalg import fractional_matrix_power

from scipy.spatial.distance import pdist


# def lsm_gaussian_kernel(x):
#     x = np.array(x)
#     nx = x.shape[0]
#     dx = x.shape[1]
#     ED = []
#     for i in range(nx):  
#         for j in range(nx):
#             xT = x[j].transpose()   #4,2
#             vecProd = np.matmul(x[i], xT)

#             SqA =  x[i]**2

#             sumSqA = np.matrix(np.sum(SqA, axis=1))
#             sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
            
#             SqB = xT**2 #4,2

#             sumSqB = np.sum(SqB, axis=0)
#             sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))  
    
#             SqED = sumSqBEx + sumSqAEx - 2*vecProd   #2,4 2,2, 2,2

#             SqED = np.sqrt(SqED)
#             #print(SqED)
#             ED.append(SqED)
#     ED = np.reshape(ED, [-1, dx, dx])
#     return ED


# a = [[1,2,3,4],[4,3,2,1]]
# b = [[[12,3,4,4],
#      [1,3,1,3]],
#      [[12,3,4,4],
#      [1,3,1,3]],
#      [[32,3,4,4],
#      [144,3,1,3]]]
     
# b = tf.convert_to_tensor(b)
# b = np.array(b)
# #c = np.matrix(c)
# print(b.shape)
# #print(b)
# #print(type(b))
# edm = lsm_gaussian_kernel(b)
# print(edm.shape)

# def lsm_cif_attention(inputs, weights, label_num):
#     #一个样例下的测试数据
#     #weights = self.attention
#     w_a = tf.zeros(weights.shape)
#     w_a_u = tf.zeros(weights.shape)
#     outs = tf.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[2])  #outs softmax layer
#     nx = inputs.shape[0]  
#     dx = inputs.shape[1]  #列
#     c = []
#     for j in range(label_num):
#         for i in range(dx):
#             w_a[i+1] = w_a[i] + weights[i]
#             if w_a[i+1]<1:
#                 outs[i+1] = outs[i] + weights[i]*inputs[i]
#             else:
#                 w_a_u[i] = 1 - w_a[i]
#                 c[j] = outs[i] + w_a_u[i] * inputs[i]  #############
#                 w_a[i] = weights[i] - w_a_u[i]
#                 outs[i] = (weights[i] - w_a_u[i])* inputs[i]    
#     return c

# def lsm(X, Y):
#     X_norm = (X ** 2).sum(1).view(-1, 1)
#     Y_norm = (Y ** 2).sum(1).view(1, -1)
#     K_tmp = X_norm + Y_norm - 2. * tf.matmul(X, tf.transpose(Y))
#     K_tmp *= -0.2
#     K = tf.exp(K_tmp)
# return K
# inputs = [[[12,3,4,4],
#         [1,3,1,3]],
#         [[12,3,4,4],
#         [1,3,1,3]]]
# inputs = tf.convert_to_tensor(inputs, dtype = float)
# x =inputs ** 2
# x = tf.reduce_sum(x, 2)
# x= tf.reshape(x, [-1, 2, 1])
# #X_norm = (x).sum(1).view(-1, 1)
# print(x)
# y = tf.reshape(x, [-1, 1, 2])
# c = x + y - 2.*tf.matmul(x, y)
# print(c) 
from sklearn.metrics import f1_score 
import numpy as np
import matplotlib.pyplot as plt
def height(x, y):
	return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)


# y_pred = [0, 1, 1, 1, 1, 1]
# y_true = [0, 1, 0, 1, 1, 1] 

# print(f1_score(y_true, y_pred, average='macro'))  
# print(f1_score(y_true, y_pred, average='weighted')) 
n = 256
# x = pd.read_csv('trX.csv')
# y = pd.read_csv('trY.csv')
# # 将原始数据变成网格数据
# X,Y = np.meshgrid(x,y)
# # 填充颜色
# plt.contourf(X,Y,height(X,Y),8,alpha=0.75,cmap=plt.cm.hot)
# # add contour lines
# C = plt.contour(X,Y,height(X,Y),8,color='black',lw=0.5)
# # 显示各等高线的数据标签cmap=plt.cm.hot
# plt.clabel(C,inline=True,fontsize=10)

# plt.show()


# x = pd.read_csv('x.csv')
# y = pd.read_csv('Y.csv')

# x = x.values
# y = y.values 
# inputs = [[[12,1,1]],[[12,1,3]],[[4,3,3]]]
# inputs = tf.convert_to_tensor(inputs, dtype = float)
# inputs = tf.reshape()
# inputs = np.array(inputs)
# print(inputs.shape)
#x = np.linspace(0, 100, 100)

#print(y[0].shape)
#print(y[0])

b = [[[12,3,4,4],
     [1,3,1,3]],
     [[12,3,4,4],
     [1,3,1,3]],
     [[32,3,4,4],
     [144,3,1,3]]]
a = tf.convert_to_tensor(b)
s = tf.reduce_sum(a,2)
print(s)
print(a)
