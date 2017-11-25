from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        # pass conv_param to the forward pass for the convolutional layer
        self.filter_size = filter_size
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        C, H, W = input_dim
        H_conv = out_dim(H, self.filter_size, self.conv_param['pad'], self.conv_param['stride'])
        W_conv = out_dim(W, self.filter_size, self.conv_param['pad'], self.conv_param['stride'])
        H_pool = out_dim(H_conv, self.pool_param['pool_height'], 0, self.pool_param['stride'])
        W_pool = out_dim(W_conv, self.pool_param['pool_width'], 0, self.pool_param['stride'])

        self.params = {
            'W1': np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size)),
            'b1': np.zeros((1, num_filters)),
            'W2': np.random.normal(scale=weight_scale, size=(num_filters * H_pool * W_pool, hidden_dim)),
            'b2': np.zeros((1, hidden_dim)),
            'W3': np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes)),
            'b3': np.zeros((1, num_classes))
        }
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # print('X', X.shape, 'W1', W1.shape, 'b1', b1.shape)
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        # print('conv_out', conv_out.shape, 'W2', W2.shape, 'b2', b2.shape)
        conv_out_p = conv_out.reshape(conv_out.shape[0], np.prod(conv_out.shape[1:]))
        # print('conv_out_p', conv_out_p.shape)
        hidden_out, hidden_cache = affine_relu_forward(conv_out_p, W2, b2)
        # print('hidden_out', hidden_out.shape, 'W3', W3.shape, 'b3', b3.shape)
        scores, scores_cache = affine_forward(hidden_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        dhidden_out, dW3, db3 = affine_backward(dscores, scores_cache)
        grads['W3'] = dW3
        grads['b3'] = db3

        dconv_out_p, dW2, db2 = affine_relu_backward(dhidden_out, hidden_cache)
        grads['W2'] = dW2
        grads['b2'] = db2
        dconv_out = dconv_out_p.reshape(conv_out.shape)

        _, dW1, db1 = conv_relu_pool_backward(dconv_out, conv_cache)
        grads['W1'] = dW1
        grads['b1'] = db1

        loss += .5 * self.reg * np.sum(np.sum(self.params[w]**2) for w in ['W1', 'W2', 'W3'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
