# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:35:55 2020

@author: 24511
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import seaborn as sns

from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, multilabel_confusion_matrix, f1_score, precision_score, recall_score, hamming_loss, f1_score  

from keras_model import lsm_network_v1, standardize, accuracy_multilabel

epochs = 100
batch_size = 1216
#加载数据
x = pd.read_csv('X.csv')
y = pd.read_csv('Y.csv')

#csv转换为numpy
x = x.values
y = y.values 
y_train = []
y_hat = []
y_id = []

#x = np.column_stack((x,y))
#x = np.column_stack((x,y))
#分割数据集,转换为tensor
x_train, x_test, Y_train, Y_test = train_test_split(x, y, test_size=1216, random_state=0)  #测试集个数大于训练集合
#得出数据尺寸
Ny, dy = Y_train.shape
Nty, dty = Y_test.shape
N = 2 * dy
batch_num = math.ceil(Ny / batch_size)
#模型
for i in range(dy):
    y_train.append(Y_train[:, i])    
y_train.append(Y_train)

#前置参数
config_parameters = {
'input_dim': x_train.shape[0],
'hidden_num': 4,
'hidden_len': [100, 50, 50, 14],
'label_num': dy,
'Y_labeled': Y_train,
'alpha':0.1
}

model = lsm_network_v1(config_parameters)

optimizer = keras.optimizers.Adam(learning_rate=0.1)
loss_metric = keras.metrics.BinaryCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy()

epochs_tqdm = tqdm(list(range(1, epochs + 1)))

best_acc = 0
loss_fig = []
acc_write = []
epoch_number = []
f1_fig = []
hl = []
reca = []
prec = []

for epoch in epochs_tqdm:
    for i in range(1, batch_num + 1):
        x_train_batch = x_train[(i-1)*batch_size:i*batch_size, :]
        x_test_batch = x_test[(i-1)*batch_size:i*batch_size, :]
        y_train_batch = [y_train[ddy][(i-1)*batch_size:i*batch_size] for ddy in range(dy)]
        Y_train_batch = Y_train[(i-1)*batch_size:i*batch_size, :]
        Y_test_batch = Y_test[(i-1)*batch_size:i*batch_size, :]
        with tf.GradientTape() as tape:
            y_train_hat, ytrainhat, y1_train_hat = model(x_train_batch)
            #loss = [mse(X_hat[v], X[v]) for v in range(V)]
            #loss = sum(loss)
            loss = 0
            for j in range(dy):
                loss += bce(y_train_batch[j], y1_train_hat[j])
            loss += bce(Y_train_batch, y1_train_hat[-1])

            #loss_edm = edm_loss(D, dy)
            #loss_rank = rank_penalty(D, target_rank=6)
            loss_sum = loss #+ loss_edm + loss_rank
            loss_fig.append(loss_sum)

        grads = tape.gradient(loss_sum, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        weight = model.trainable_variables
        # model.save_weights('my_model.h5')
        # model.load_weights('my_model.h5')
        #print(weight)

        new_y_test_hat, newytesthat, new_y1_test_hat = model(x_test_batch)
        new_Y_test_hat = new_y1_test_hat[-1]
        new_Y_test_hat = np.array(new_Y_test_hat)

        new_Y_test_hat = standardize(new_Y_test_hat)

        #print(new_Y_test_hat.shape)
        
        new_Y_test_hat[new_Y_test_hat >= 0.5] = 1
        new_Y_test_hat[new_Y_test_hat < 0.5] = 0
        new_Y_test_hat = np.array(new_Y_test_hat)

        #print(new_Y_test_hat.shape)
        #print(Y_test_batch)
        acc = accuracy_multilabel(Y_test_batch, new_Y_test_hat)
        hammingloss = hamming_loss(Y_test_batch, new_Y_test_hat)
        hl.append(hammingloss)
        f1 = f1_score(Y_test_batch, new_Y_test_hat, average='samples')
        pre = precision_score(Y_test_batch, new_Y_test_hat, average='samples')
        prec.append(pre)
        recall = recall_score(Y_test_batch, new_Y_test_hat, average='samples')
        reca.append(recall)
        f1_fig.append(f1)
        #f1 = f1_score(Y_test_batch, new_Y_test_hat)
        acc_write.append(acc)
        if acc > best_acc:
            best_acc = acc
            model.save_weights('my_model.h5')
            # model.load_weights('my_model.h5')
        epochs_tqdm.set_description("||epoch %s||mse: %f||acc: %f||best_acc: %f||Hamming Loss: %f||f1: %f||pre: %f||rec: %f||" % (epoch, loss_metric(Y_test_batch, new_Y_test_hat), acc, best_acc,hammingloss, f1, pre, recall))
        epoch_number.append(epoch)
    
fig = plt.figure(figsize = (10,5))       #figsize是图片的大小`
fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.5, hspace=0.5)
plt.subplot(231)
p1 = plt.plot(epoch_number, acc_write, 'bo--')
plt.title('The accuracy of the emotions')
plt.subplot(232)
p2 = plt.plot(epoch_number, loss_fig, 'r-')
plt.title('The total loss of the emotions')
plt.subplot(233)
p3 = plt.plot(epoch_number, f1_fig, 'b-')
plt.title('The f1 score of the emotions')
plt.subplot(234)
p4 = plt.plot(epoch_number, hl, 'g-')
plt.title('The hamming loss of the emotions')
plt.subplot(235)
p5 = plt.plot(epoch_number, prec, 'o-')
plt.title('The precision of the emotions')
plt.subplot(236)
p6 = plt.plot(epoch_number, reca, 'y-')
plt.title('The recall of the emotions')
plt.savefig('result01.png')

y_hat.append(new_Y_test_hat)
y_id.append(new_y_test_hat)
y_id = np.reshape(y_id, [-1, dy])
#print(y_hat)
y_hat=np.reshape(y_hat,[-1, dy])
np.savetxt('acc.csv',acc_write)
np.savetxt("loss.csv", loss_fig)
np.savetxt('f1.csv',f1_fig)
np.savetxt('hl.csv',hl)
np.savetxt('pre.csv',prec)
np.savetxt('rec.csv',reca)

