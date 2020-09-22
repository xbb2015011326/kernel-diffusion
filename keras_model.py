import numpy as np
import scipy
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix
from keras_kernel import infer

import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model

class lsm_network_v1(Model):
    def __init__(self, config_parameters):
        super(lsm_network_v1, self).__init__()
        self.input_dim = config_parameters['input_dim']
        self.hidden_num = config_parameters['hidden_num']
        self.hidden_len = config_parameters['hidden_len']
        self.label_num = config_parameters['label_num']
        self.Y_labeled = config_parameters['Y_labeled'] 
        self.alpha = config_parameters['alpha']
        self.dense1 = layers.Dense(self.hidden_len[0], activation='relu')
        
        self.layer = {}
        for i in range(self.label_num):
            self.layer['output_{}'.format(i)] = layers.Dense(2, activation='sigmoid')
            for j in range(self.hidden_num):
                self.layer['label{}_layer{}'.format(i, j)] = layers.Dense(self.hidden_len[j], activation='relu')
        
        self.layer['K'] = layers.Dense(self.hidden_len[-1], activation='tanh', name='K')
        self.layer['Dot'] = layers.Dense(self.hidden_len[-1], use_bias=False)
        self.layer['ouput_final'] = layers.Dense(self.label_num, activation='softmax', name='outputs')
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(1)
        self.dot = layers.Dot((1, 1))

          
    def call(self, x):
        alpha, input_dim, hidden_num, hidden_len, label_num = self.alpha, self.input_dim, self.hidden_num, self.hidden_len, self.label_num
        y_train = self.Y_labeled

        Z_latent = []  #隐藏层列表
        h_latent = {}  #前向网络字典
        mu_new = []
        # build model forward
        for i in range(label_num):
            for j in range(hidden_num):   
                if j == 0:
                    h_latent['label{}_layer{}'.format(i, j)] = self.layer['label{}_layer{}'.format(i, j)](x)
                elif j == hidden_num - 1:
                    Z_latent.append(self.layer['label{}_layer{}'.format(i, j)](h_latent['label{}_layer{}'.format(i, j-1)]))                  
                else:
                    h_latent['label{}_layer{}'.format(i, j)] = self.layer['label{}_layer{}'.format(i, j)](h_latent['label{}_layer{}'.format(i, j - 1)])

        y1_hat = [self.layer['output_{}'.format(i)](Z_latent[i])[:, 0] for i in range(label_num)]
        
        y_hat = np.array(y1_hat)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0 
         
        #print(Z_latent)                                             
        l_Z_latent = tf.concat(tf.reshape(Z_latent, [-1, label_num, hidden_len[hidden_num-1]]), axis=1)   #  1200,7,7 
        #kernel matrix based on rbf
        l_z_new = tf.transpose(l_Z_latent,[0, 2, 1])
        #l_z_new = np.array(l_z_new)
        l_Z_latent_norm = tf.reshape(tf.reduce_sum(l_z_new, 2), [-1, label_num, 1])
        #l_new_norm = tf.reshape(tf.reduce_sum(l_z_new, 2), [-1, 1, label_num])
        for i in range(1):
            parser = argparse.ArgumentParser(description='Initialize trainning parameters which are "mode" and "iter_size".')
            parser.add_argument('-mode', type=str, default = 'dnn', help='you can choose kernel mode from "dnn" or "rbf". Default is "dnn".')
            parser.add_argument('-iter_size', type=int, default=100, help='iter size of mcmc.')
            parser.add_argument('-x', type=float, default = l_Z_latent_norm[i], help='inputs')
            parser.add_argument('-y',type=float, default = y_train[i], help='outputs' )
            args = parser.parse_args()
            assert args.mode in ['dnn', 'rbf'], f'{args.mode} mode is not implemented. please choose from "dnn" or "rbf"'
            mu, sigma2 = infer(l_Z_latent_norm[i], y_train[i], args.mode, args.iter_size)
            mu = tf.convert_to_tensor(mu)
            mu_new.append(mu)

        mu_new = tf.concat(tf.reshape(mu_new,[-1, label_num, label_num]), axis=0)
        print(mu)  
        l_new_norm = tf.reshape(tf.reduce_sum(mu_new, 2), [-1, 1, label_num])

        K_tmp = l_Z_latent_norm  + l_new_norm - 2. * tf.matmul(l_Z_latent_norm, l_new_norm)
        gamma = 1.0 / l_z_new.shape[2]
        K_tmp *= 0 - gamma
        K = tf.exp(K_tmp)
        print(K.shape)
        print(K)
        #normalize the kernel stored
        S = tf.reduce_sum(K, axis=1)

        w1n = np.array(1./np.sqrt(S))
        w1n = tf.linalg.diag_part(tf.reshape(w1n, [label_num, -1, 1]))

        Wn = w1n * K *w1n
        #Initiliaze the y vector for each class
        #Z = tf.zeros((l_Z_latent.shape[0], label_num, 1))  
        A = tf.eye(Wn.shape[1], Wn.shape[2], [Wn.shape[0]]) - alpha * Wn
        A = np.array(A)
        #A = np.reshape(A, [-1, A.shape[1]*A.shape[2]])
        y = np.reshape(y_hat, [-1, label_num, 1])
        pro_z = []
        for j in range(A.shape[0]):
            for a in range(A.shape[1]):
                A[:, a, a] = A[:, a, a] + 1e-6
            f, _ = scipy.sparse.linalg.cg(A[j, :, :], y[j, :, :], tol=1e-6, maxiter=30)  #7,1
            pro_z.append(f)
        #print(pro_z)
        #pro_z = np.array(pro_z)
        pro_z = tf.reshape(pro_z, [-1, label_num, 1])
        #print(pro_z)
        #cif-attention
        yhat = np.array(pro_z)
        yhat[yhat >= 0.5] = 1
        yhat[yhat < 0.5] = 0


        attention = self.layer['K'](l_Z_latent)
        attention = self.flatten(self.dense_1(attention))
        attention_weight = tf.nn.softmax(attention)
        attention_output = self.dot([l_Z_latent, attention_weight])

        #print(score)
        Y_pred = self.layer['ouput_final'](attention_output)   #y_hat  none, 7, 1
        #print(Y_pred.shape)
        # y2_hat = [tf.reshape(Y_pred[:, l, :], [-1, 1]) for l in range(label_num)] 
        #print(y2_hat)
        #print(Y_pred.shape)
        y1_hat.append(Y_pred)
        return y_hat, yhat, y1_hat

def standardize(x):
    return (x - np.mean(x)) / (np.std(x)) 

def accuracy_multilabel(y_true, y_pred):

    Ny, dy = y_true.shape
    Cm = multilabel_confusion_matrix(y_true, y_pred)
    accuracy = 0
            
    #print(Cm.shape)

    for i in range(dy):
        accuracy += (Cm[i, 0, 0] + Cm[i, 1, 1]) / (Ny)
    accuracy = accuracy / dy
    return accuracy


