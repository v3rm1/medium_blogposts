'''
Linear regression sample with tensorflow.

Author: v3rm1

'''

from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
random_num_gen = np.random


# Model parameters
learning_rate = 0.01
training_epochs = 1000
log_steps = 10
log_dir = "./logdir"
start_time = int(round(time.time() * 1000))

# Training Data: Linear (y = 2.019x + 25)
lin_X = np.asarray(
[0.1,
 0.7,
 1.3,
 1.9,
 2.5,
 3.1,
 3.7,
 4.3,
 4.9,
 5.5,
 6.1,
 6.7,
 7.3,
 7.9,
 8.5,
 9.1,
 9.7,
 10.3,
 10.9,
 11.5],
)
lin_Y = np.asarray(
[25.2019,
 26.4133,
 27.6247,
 28.8361,
 30.0475,
 31.2589,
 32.4703,
 33.6817,
 34.8931,
 36.1045,
 37.3159,
 38.5273,
 39.7387,
 40.9501,
 42.1615,
 43.3729,
 44.5843,
 45.7957,
 47.0071,
 48.2185],
)
num_samples = lin_X.shape[0]

# Starting an interactive tensorflow session
sess = tf.InteractiveSession()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6007")
# Tensorflow Graph Input Layer
with tf.name_scope('data') as scope:
    X = tf.placeholder("float")
    tf.summary.scalar('X', X)
    Y = tf.placeholder("float")
    tf.summary.scalar('Y', Y)

# Model Weights and Biases
with tf.name_scope('model') as scope:
    W = tf.Variable(random_num_gen.randn(), name="weight")
    tf.summary.scalar('W', W)
    b = tf.Variable(random_num_gen.randn(), name="bias")
    tf.summary.scalar('b', b)

# Linear Model Constructor
with tf.name_scope('linear_model') as scope:
    pred_Y = tf.add(tf.multiply(W, X), b)
    tf.summary.scalar('pred_Y', pred_Y)

# Cost function: Mean Squared Error
with tf.name_scope('cost') as scope:
    cost = tf.reduce_sum(tf.pow(pred_Y, 2)/(2*num_samples))
    tf.summary.scalar('cost', cost)

# Training function: Gradient Descent
grad_desc = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Variable initialization function
init_var = tf.global_variables_initializer()

# Create a model saver instance to allow saving and retrieval of models
saver = tf.train.Saver()

# Session Variable: Session creation
#with tf.Session() as sess:
    # TODO: RESTRUCTURE CODE TO USE THE DEBUGGER: LEARNING PURPOSE ONLY
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# Summary writer
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir, sess.graph)

# Initialize variables in session
sess.run(init_var)

    # Training process
for epoch in range(training_epochs):
    # running grad_desc function for training epochs
    for (x, y) in zip(lin_X, lin_Y):
        summary, accuracy = sess.run([merged, grad_desc], feed_dict={X:x, Y:y})

    # printing logs per epoch
    if (epoch + 1) % log_steps == 0:
        cost_update = sess.run(cost, feed_dict={X: lin_X, Y: lin_Y})
        writer.add_summary(summary, epoch+1)
        print("Epoch Num: {0:4d}\nCost: {1:2.9f}\nWeight: {2:3.9f}\t\tBias: {3:3.9f}".format(epoch+1, cost_update, sess.run(W), sess.run(b)))

save_path = "./model/lin_reg_" + time.strftime("%Y%m%d-%H%M%S") + ".ckpt"
save_model = saver.save(sess, save_path)
print("Training process complete.\n Model saved at: {0}".format(save_path))


end_time = int(round(time.time() * 1000))
print("Process complete. Time taken: {0:10.9f}".format(end_time - start_time))
