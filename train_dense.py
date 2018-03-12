'''
Deep Learning Programming
--------------------------------------
Name: Vishal Tomar
Roll No.: 14CS30038

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import tensorflow as tf
import numpy as np
import os
    
RANDOM_SEED = 59
tf.set_random_seed(RANDOM_SEED)
n_classes = 2

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    # Layer's sizes
    x_size = trainX.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = trainY.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, 1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        hm_epochs = 0
        batch_size = 100
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for k in xrange(0,trainX[:,0].size,batch_size):
                epoch_x,epoch_y = trainX[k:k+batch_size,:], trainY[k:k+batch_size,:]

                sess.run(updates, feed_dict={X: epoch_x, y: epoch_y})

            train_accuracy = np.mean(np.argmax(trainY, axis=1) ==
                                 sess.run(predict, feed_dict={X: trainX, y: trainY}))

            print("Epoch = %d, train accuracy = %.2f%% "  % (epoch + 1, 100. * train_accuracy))
    
        W0 = w_1.eval()
        W1 = w_2.eval()
        np.savez('weights_dense.npz', a=W0, b=W1)


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''        

    # Layer's sizes
    x_size = testX.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, n_classes])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, n_classes))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, 1)
    
    with tf.Session() as sess:
        data = np.load('weights_dense.npz')
        W0 = data['a']
        W1 = data['b']
        assign_op = w_1.assign(W0)
        sess.run(assign_op)
        assign_op = w_2.assign(W1)
        sess.run(assign_op)
        labels = sess.run(predict, feed_dict={X: testX})
        
    return labels
