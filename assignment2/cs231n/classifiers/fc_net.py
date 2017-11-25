from builtins import range
from builtins import object
import numpy as np
import functools

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params = {
            'W1': np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim)),
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes)),
            'b2': np.zeros((1, num_classes))
        }
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        l1_out, l1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        l2_out, l2_cache = affine_forward(l1_out, self.params['W2'], self.params['b2'])
        scores = l2_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(l2_out, y)

        l2_dout, l2_dw, l2_db = affine_backward(dscores, l2_cache)
        grads['W2'] = l2_dw + self.reg * self.params['W2']
        grads['b2'] = l2_db

        l1_dout, l1_dw, l1_db = affine_relu_backward(l2_dout, l1_cache)
        grads['W1'] = l1_dw + self.reg * self.params['W1']
        grads['b1'] = l1_db

        loss += .5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    @classmethod
    def format_param(cls, name, i):
        return '{}{}'.format(name, str(i + 1))

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        for i, (D, M) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[self.format_param('W', i)] = np.random.normal(scale=weight_scale, size=(D, M))
            self.params[self.format_param('b', i)] = np.zeros((1, M))

        if self.use_batchnorm:
            for i, M in enumerate(hidden_dims):
                self.params[self.format_param('gamma', i)] = np.ones((1, M))
                self.params[self.format_param('beta', i)] = np.zeros((1, M))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)] if self.use_batchnorm else []

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def get_affine_params(self, i):
        return self.params[self.format_param('W', i)], self.params[self.format_param('b', i)]


    def set_gradient_params(self, grads, dw, db, dgamma, dbeta, i):
        w_key = self.format_param('W', i)
        grads[w_key] = dw + self.reg * self.params[w_key]

        b_key = self.format_param('b', i)
        grads[b_key] = db

        if dgamma is not None:
            gamma_key = self.format_param('gamma', i)
            grads[gamma_key] = dgamma

        if dbeta is not None:
            beta_key = self.format_param('beta', i)
            grads[beta_key] = dbeta


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

        def compute_forward_layer(acc, i):
            x, out = acc
            w, b = self.get_affine_params(i)
            x, affine_cache = affine_forward(x, w, b)

            batchnorm_cache = None
            if self.use_batchnorm:
                gamma = self.params[self.format_param('gamma', i)]
                beta = self.params[self.format_param('beta', i)]
                x, batchnorm_cache = batchnorm_forward(x, gamma, beta, self.bn_params[i])

            x, relu_cache = relu_forward(x)

            dropout_cache = None
            if self.use_dropout:
                x, dropout_cache = dropout_forward(x, self.dropout_param)

            out.append((affine_cache, batchnorm_cache, relu_cache, dropout_cache))
            return x, out

        conv_out, caches = functools.reduce(compute_forward_layer, range(self.num_layers - 1), (X, []))

        L = self.num_layers - 1
        w, b = self.get_affine_params(L)

        scores, cache_final = affine_forward(conv_out, w, b)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

        dconv_out, dw, db = affine_backward(dscores, cache_final)
        self.set_gradient_params(grads, dw, db, None, None, L)

        dx = dconv_out
        for i, cache in reversed(list(enumerate(caches))):
            affine_cache, batchnorm_cache, relu_cache, dropout_cache = cache

            if self.use_dropout:
                dx = dropout_backward(dx, dropout_cache)

            dx = relu_backward(dx, relu_cache)

            dgamma, dbeta = None, None
            if self.use_batchnorm:
                dx, dgamma, dbeta = batchnorm_backward(dx, batchnorm_cache)

            dx, dw, db = affine_backward(dx, affine_cache)
            self.set_gradient_params(grads, dw, db, dgamma, dbeta, i)

        for i in range(self.num_layers):
            w_key = self.format_param('W', i)
            loss += .5 * self.reg * (np.sum(self.params[w_key]**2))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
