import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    correct_class_score = scores[y[i]]
    socres_exp_sum = np.sum(np.exp(scores))
    loss += -correct_class_score + np.log(socres_exp_sum)

    for j in xrange(num_classes):
      dW[:, j] += np.exp(scores[j]) / socres_exp_sum * X[i]
    dW[:, y[i]] -= X[i]

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W) / 2.0
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = np.dot(X, W)
  correct_class_score = scores[range(num_train), y].reshape(num_train, 1)
  socres_exp_sum = np.sum(np.exp(scores), axis = 1, keepdims = True)
  loss += np.sum(-correct_class_score + np.log(socres_exp_sum))

  dz = np.exp(scores) / socres_exp_sum
  dz[range(num_train), y] -= 1
  dW = np.dot(X.T, dz)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W) / 2.0
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

