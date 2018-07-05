from __future__ import print_function

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# GPU setting
import os
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from tensorflow.examples.tutorials.mnist import input_data

# hyper paameter setting
image_size = 28
batch_size = 64
valid_size = test_size = 10000
num_data_input = image_size * image_size
num_hidden = 100
num_labels = 10
dataset = 'mnist'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels
filename = "mnist_EDL.csv"
iterat_kappa = 1
act_f = "relu"  # Activation function
# Initializing weights using init_f distribution. Recommend to use uniform.
init_f = "uniform"
# Initializing direct feed back weights using back_init_f distribution. Recommend to use uniform.
back_init_f = "uniform"

weight_uni_range = 0.01
back_uni_range = 0.004
lr = 0.001
num_layer = 4# should be >= 3
num_steps = 60001
train_dataset = X_train
train_labels = y_train
valid_dataset = X_test
valid_labels = y_test
test_dataset = X_test
test_labels = y_test


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def sample_Z(X, m, n, kappa):
    return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    # return (X + np.random.normal(0, kappa, size=[m, n]))

def drelu(x):
    zero = tf.zeros(x.get_shape())
    one = tf.ones(x.get_shape())
    return(tf.where(tf.greater(x, zero), one, zero))

def dtanh(x):
    return(1-tf.multiply(tf.nn.tanh(x),tf.nn.tanh(x)))

def dsigmoid(x):
    return( tf.multiply((1-tf.nn.sigmoid(x)), tf.nn.sigmoid(x)))

def act_ftn(name):
    if(name == "tanh"):
        return(tf.nn.tanh)
    elif(name == "relu"):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)
    else:
        print("not tanh or relu")
        
def dact_ftn(name):
    if(name == "tanh"):
        return(dtanh)
    elif(name == "relu"):
        return(drelu)
    elif(name == "sigmoid"):
        return(dsigmoid)
    else:
        print("not tanh or relu")

def init_ftn(name, num_input, num_output, runiform_range):
    if(name == "normal"):
        return(tf.truncated_normal([num_input, num_output]))
    elif(name == "uniform"):
        return(tf.random_uniform([num_input, num_output], minval = -runiform_range, maxval = runiform_range ))
    else:
        print("not normal or uniform")


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels



class Weights:
    def __init__(self, batch_size, num_input, num_output, num_final,
                 act_f, init_f, notfinal=True, back_init_f="uniform",
                 weight_uni_range=0.05, back_uni_range=1):
        self.weights = tf.Variable(
            init_ftn(init_f, num_input, num_output, weight_uni_range))
        self.biases = tf.Variable(tf.zeros([num_output]))
        backward_t = tf.Variable(
            init_ftn(back_init_f, num_final, num_output, back_uni_range))
        self.backward = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                   batch_size, num_final, num_output])
        self.batch_size = batch_size
        self.num_input = num_input
        self.num_output = num_output
        self.num_final = num_final
        self.activation = act_ftn(act_f)
        self.dactivation = dact_ftn(act_f)
        self.notfinal = notfinal
        self.inputs = None
        self.before_activation = None


    def __call__(self, x, batch_size):
        if (batch_size == self.batch_size):
            self.inputs = tf.reshape(x, [batch_size, self.num_input, 1])
            self.before_activation = tf.matmul(x, self.weights) + self.biases
            if (self.notfinal):
                return(self.activation(self.before_activation))
            else:
                return(self.before_activation)
        else:
            before_activation = tf.matmul(x, self.weights) + self.biases
            if (self.notfinal):
                return(self.activation(before_activation))
            else:
                return(before_activation)

    def optimize(self, dError_dy, cost, lr=0.01):
        #dError_dy dim is [batch_size, 1, num_fianl]
        if (self.notfinal):
            # Fix the feedback
            # grad_list  = [tf.gradients(output[:, idx], layer_out)[0] for idx in range(out_num_nodes)]
            # l_grad     = tf.transpose(tf.reduce_mean(tf.stack(grad_list), 1))
            
            ## 1
            dError_dhidden = tf.matmul(dError_dy,self.backward)
            
            ## 2
            # dError_dhidden = tf.matmul(dError_dy,\
            # tf.matmul(self.backward, tf.matrix_diag(self.dactivation(self.before_activation))))
            delta_weights = tf.reduce_mean(\
            tf.matmul(self.inputs, dError_dhidden), 0) + 0.01*self.weights
            delta_biases = tf.reduce_mean(\
            dError_dhidden, 0) + 0.01*self.biases
        else:
            delta_weights = tf.reduce_mean(
                tf.matmul(self.inputs, dError_dy), 0) + 0.01*self.weights
            delta_biases = tf.reduce_mean(dError_dy, 0) + 0.01*self.biases

        # delta_weights = tf.truediv(delta_weights,1+tf.norm(delta_weights))
        # delta_biases= tf.truediv(delta_biases,1+tf.norm(delta_biases))
        change_weights = tf.assign_sub(self.weights, lr * delta_weights)
        change_biases  = tf.assign_sub(self.biases, lr * tf.reshape(delta_biases, (self.num_output,)))
        return change_weights, change_biases

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # model building
    Weight_list = {}
    name = "W0"
    Weight_list[name] = Weights(batch_size, num_data_input, num_hidden, num_labels,
                                act_f, init_f, True, back_init_f, weight_uni_range, back_uni_range)
    for i in range(num_layer - 3):
        name = "W" + str(i + 1)
        Weight_list[name] = Weights(batch_size, num_hidden, num_hidden, num_labels,
                                    act_f, init_f, True, back_init_f, weight_uni_range, back_uni_range)
    name = "W" + str(num_layer - 2)
    Weight_list[name] = Weights(batch_size, num_hidden, num_labels, num_labels,
                                act_f, init_f, False, back_init_f, weight_uni_range, back_uni_range)
    y_train = None
    x_train = tf_train_dataset

    # Train predictions
    for i in range(num_layer - 1):
        name = "W" + str(i)
        if (i != num_layer - 2):
            x_train = Weight_list[name](x_train, batch_size)
        else:
            y_train = Weight_list[name](x_train, batch_size)
    logits = y_train
    
    
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = tf_train_labels)
    loss = tf.reduce_mean(cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    dError_dy = tf.reshape(tf.gradients(cross_entropy, logits)[
                           0], [batch_size, 1, num_labels])
    dec = tf.train.exponential_decay(lr, global_step, 100000, 0.99, staircase=True)

    # Training optimization
    train_list = []
    for i in range(num_layer - 1):
        name = "W" + str(i)
        train_list += Weight_list[name].optimize(dError_dy, cross_entropy, dec)


    # Validation Phase
    y_valid = None
    x_valid = tf_valid_dataset
    for i in range(num_layer - 1):
        name = "W" + str(i)
        if (i != num_layer - 2):
            x_valid = Weight_list[name](x_valid, valid_size)
        else:
            y_valid = Weight_list[name](x_valid, valid_size)
    logits_valid = y_valid

    # Test
    y_test = None
    x_test = tf_test_dataset
    for i in range(num_layer - 1):
        name = "W" + str(i)
        if (i != num_layer - 2):
            x_test = Weight_list[name](x_test, test_size)
        else:
            y_test = Weight_list[name](x_test, test_size)
    logits_test = y_test

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(logits_valid)
    test_prediction = tf.nn.softmax(logits_test)


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")

    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]

      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

      # Optimization call
      # For loss and predictions
      l, predictions = session.run(
          [loss, train_prediction], feed_dict=feed_dict)

      # For the training part
      session.run(train_list, feed_dict=feed_dict)
      
      batch_noise = sample_Z(
      batch_data, batch_data.shape[0], batch_data.shape[1], 1)
    # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset: batch_noise, tf_train_labels: batch_labels}
      # For the training part
      session.run(train_list, feed_dict=feed_dict)
      
      if (step % 500 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.1f%%" %
              accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(
        test_prediction.eval(), test_labels))

