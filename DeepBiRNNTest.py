

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import Mondaq data
from preprocessing2 import Items
import time
from astropy.table import Table

import math 



entiredata = Items([],[])


trainset, testset = entiredata.splitTrainTest()



start_time = time.time()


learning_rate = 0.0001
training_iters = 2000000
batch_size = 128

display_step = 10


n_input = 1 
n_steps = 12 
n_hidden = 160 
n_classes = 29 
number_of_layers = 5 


x = tf.placeholder("float", [None, n_steps, 1])
y = tf.placeholder("float", [None, n_classes])


weights = {

    'out': tf.Variable(tf.random_uniform([2*n_hidden, n_classes], minval=-math.sqrt(6/13), maxval= math.sqrt(6/13)))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):

    x = tf.unstack(x, n_steps, 1)

    forwardCells = []
    backwardCells = []
    for _ in range(number_of_layers):

        lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

        lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

        forwardCells.append(lstm_fw_cell)
        backwardCells.append(lstm_bw_cell)

    stacked_forwardlstm = tf.contrib.rnn.MultiRNNCell(forwardCells, state_is_tuple=True)
    stacked_backwardlstm = tf.contrib.rnn.MultiRNNCell(backwardCells, state_is_tuple=True)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)


    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(stacked_forwardlstm, stacked_backwardlstm, x,
                                              dtype=tf.float32)
    except Exception: 
        outputs = rnn.static_bidirectional_rnn(stacked_forwardlstm, stacked_backwardlstm, x,
                                        dtype=tf.float32)
 
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


print("--- %s seconds ---" % (time.time() - start_time))

