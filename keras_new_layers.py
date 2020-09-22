import math
import numpy as np
import scipy

import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model

class lsm_Graph_adj(Model):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, features, output, bias=True, init='uniform'):
        super(lsm_Graph_adj, self).__init__()
        self.input = features
        self.output = output
        self.weight = tf.compat.v1.get_variable(name="self.weight", 
                                 shape=[features[0], features[1]], 
                                 initializer=tf.constant_initializer(np.array(features)), 
                                 trainable=False
                                )
        if bias:
            self.bias = tf.compat.v1.get_variable(name="self.bias", 
                                 shape=[output[0], output[1]], 
                                 initializer=tf.constant_initializer(np.array(output)), 
                                 trainable=False
                                )
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = tf.matmul(input, self.weight)
        output = tf.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input) + ' -> ' \
               + str(self.output) + ')'

class dist2SimiMatrix_scale_function(Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        r,d = input.size()
        scale = weights
        x = F.relu(input)
        ONE = torch.ones(r,1).cuda()
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        y = torch.exp(-scale * xv.mm(x).mm(vx) * r)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, weights = ctx.saved_variables
        r, d = input.size()
        scale = weights
        x = F.relu(input)
        ONE = torch.ones(r, 1)
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        dzds = -scale * dzdy * r * dzdy
        y = -0.5 * (dzds * (xv.pow(3).mm(x).mm(vx)).mm(ONE).mm(ONE.transpose(0,1))) - 0.5 * ONE * ONE.transpose(1, 0).mm(dzds * (xv.mm(x).mm(vx.pow(3)))) + xv.mm(dzds.mm(vx))
        dw = (-dzdy * (xv.mm(x).mm(vx)) * r).sum()
        return y, dw