import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist

def poly_kernel(X, Y=None, degree=3, gamma=None, coef0=1):

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K_tmp = tf.matmul(X, tf.transpose(Y))  #x or hidden features?
    K_tmp *= gamma
    K_tmp += coef0
    K = K_tmp ** degree

    return K


def rbf_kernel(X, Y=None, gamma=None):

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    K_tmp = X_norm + Y_norm - 2. * tf.matmul(X, tf.transpose(Y))
    K_tmp *= 0 - gamma
    K = tf.exp(K_tmp)

    return K

def lsm_eudiStance(x):
    x = np.array(x)
    nx = x.shape[0]
    dx = x.shape[1]
    ED = []
    for i in range(nx):  
        for j in range(nx):
            xT = x[j].transpose()   #4,2
            vecProd = np.matmul(x[i], xT)

            SqA =  x[i]**2

            sumSqA = np.matrix(np.sum(SqA, axis=1))
            sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
            
            SqB = xT**2 #4,2

            sumSqB = np.sum(SqB, axis=0)
            sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))  
    
            SqED = sumSqBEx + sumSqAEx - 2*vecProd   #2,4 2,2, 2,2

            SqED = np.sqrt(SqED)
            #print(SqED)
            ED.append(SqED)
    ED = np.reshape(ED, [-1, dx, dx])
    return ED


