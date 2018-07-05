


########################### New Try 

# Deep Learning Simulations
# The class file
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
# import os
# import sys
import random
# import time
import numpy as np
# from sklearn import preprocessing
import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
import operator
from functools import reduce


####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################
def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1 * np.sqrt(4.0 / (fan_in + fan_out))
   high = 1 * np.sqrt(4.0 / (fan_in + fan_out))
   return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)


#############################################################################################
def weight_variable(shape, trainable, name):
   initial = xavier(shape[0], shape[1])
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
# Bias function
def bias_variable(shape, trainable, name):
   initial = tf.random_normal(shape, trainable, stddev=1)
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
#  Summaries for the variables
def variable_summaries(var, key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries' + key):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev' + key):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))


#############################################################################################
# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.reward = []
        self.Evaluation = {}
        self.Summaries = {}
        self.keys = []
        self.preactivations = []
        self.sess = tf.Session(config=config)

#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = weight_variable(
                    [input_dim, output_dim], trainable=trainability, name='Weight' + key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = bias_variable(
                    [output_dim], trainable=trainability, name='Bias' + key)
            with tf.name_scope('Wx_plus_b' + key):
                preactivate = tf.matmul(
                    input_tensor, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
                self.preactivations.append(preactivate)
            activations = act(preactivate, name='activation' + key)
        return activations


#############################################################################################
    def Custom_Optimizer(self, lr):
        temp   = tf.gradients( (self.classifier['cost_NN']), self.keys)
        a = []
        for i, e in enumerate(temp):
            a.append(e)
        b = [(tf.placeholder("float32", shape=grad.get_shape())) for grad in a]
        var_list = [item for item in self.keys]
        c = tf.train.AdamOptimizer(0.01).apply_gradients([(e, var_list[i]) for i, e in enumerate(b)])
        return a, b, c


#############################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, par='GDR'):
        with tf.name_scope("FLearners_1"):            #Setup initial noise for the system
            # initial = tf.random_uniform([1, Layers[0]], minval=-1, maxval=1, dtype=tf.float32)
            initial = tf.random_normal([1, Layers[0]], mean=0.0, stddev=1)
            #Declare the noise variable
            self.Deep['noise'] = tf.Variable(
                initial, trainable=False, name='noise')

            self.classifier["learning_rate"] = tf.placeholder(
            tf.float32, [], name='learning_rate')

###################### Setting up the network
            # First layer of the neural network
            i = 1
            self.Deep['FL_layer_10'] = tf.placeholder(
                tf.float32, shape=[None, Layers[0]])
            input_noise = self.Deep['FL_layer_10'] + self.Deep['noise']
        
            self.Deep['FL_layer_11'] = self.nn_layer(
                input_noise, Layers[i - 1], Layers[i], act=act_function, trainability=False, key='FL_layer_1' + str(i))
            
            # Put all the keys for the first layer in an array
            self.keys.append(self.classifier['Weight' + 'FL_layer_1' + str(i)])
            self.keys.append(self.classifier['Bias' + 'FL_layer_1' + str(i)])

            # Neural network for the rest of the layers.
            for i in range(2, len(Layers)):
                self.Deep['FL_layer_1' + str(i)] = self.nn_layer(self.Deep['FL_layer_1' + str(i - 1)], Layers[i - 1],
                Layers[i], act=act_function, trainability=False, key='FL_layer_1' + str(i))
                self.keys.append(
                    self.classifier['Weight' + 'FL_layer_1' + str(i)])
                self.keys.append(
                    self.classifier['Bias' + 'FL_layer_1' + str(i)])

        with tf.name_scope("Classifier"):
            self.classifier['class_1'] = self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
            self.keys.append(self.classifier['Weightclass'])
            self.keys.append(self.classifier['Biasclass'])
    
############ The network without the noise, primarily for output estimation
        self.Deep['FL_layer_2' + str(0)] = self.Deep['FL_layer_10']
        for i in range(1, len(Layers)):
            key = 'FL_layer_1' + str(i)
            preactivate = tf.matmul(self.Deep['FL_layer_2' + str(i - 1)],\
            self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            self.Deep['FL_layer_2' +
                      str(i)] = act_function(preactivate, name='activation_2' + key)

        self.classifier['class_2'] = tf.identity(tf.matmul(self.Deep['FL_layer_2' + str(
            len(Layers)-1)], self.classifier['Weightclass']) + self.classifier['Biasclass'])
        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

########## Design the trainer
        with tf.name_scope("Trainer"): 
            # self.classifier["Error_Loss"] = tf.nn.l2_loss(
            #     self.classifier['class_1'] - self.classifier['Target'])
            # self.classifier["Noise_Loss"] = 1e-4* \
            #     tf.reduce_mean(tf.nn.l2_loss(self.Deep['noise']))

            # Error_Loss
            self.classifier["Error_Loss"] =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
            logits= self.classifier['class_1'], labels = self.classifier['Target'], name='Error_Cost'))
            #self.classifier["Noise_Loss"] = tf.reduce_mean( tf.log(tf.sigmoid(self.Deep['noise'])))
            self.classifier["Noise_Loss"]  = 1e-04* tf.reduce_mean(tf.nn.l2_loss(self.Deep['noise']))
            Reg = 0
            for element in self.keys:
                Reg = Reg + tf.nn.l2_loss(element)

            # Total Loss
            self.classifier["cost_NN"] = self.classifier["Error_Loss"]-\
            self.classifier["Noise_Loss"] + 1e-05 * Reg
            
            # Call the optimizer
            self.Trainer["grads"], self.Trainer["grad_placeholder"],\
            self.Trainer["apply_placeholder_op"] =\
            self.Custom_Optimizer(self.classifier["learning_rate"])

            # sig_grad = tf.sign(  tf.reduce_mean\
            # (tf.gradients(self.classifier["cost_NN"], self.classifier['class_1'])))
            # self.reward['cum'] = tf.placeholder( tf.float32, [])

################# Decay the learning rate and apply the noise operation
            # Train Noise
            self.Trainer["noise_grad"] = tf.gradients(self.classifier['cost_NN'], self.Deep['noise'])
            b = [(-1*grad) for grad in self.Trainer["noise_grad"]]
            var_list = [self.Deep["noise"]]
            self.Trainer["apply_noise_op"] = \
                tf.train.AdamOptimizer(0.01).apply_gradients(
                [(e, var_list[i]) for i, e in enumerate(b)])

############## The evaluation section of the methodology
        with tf.name_scope('Evaluation'):
            with tf.name_scope('CorrectPrediction'):
                self.Evaluation['correct_prediction'] = \
                tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_2']), 1),\
                tf.argmax(self.classifier['Target'], 1))

            with tf.name_scope('Accuracy'):
                self.Evaluation['accuracy'] = tf.reduce_mean(
                    tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            with tf.name_scope('Prob'):
                self.Evaluation['prob'] = tf.cast(
                    tf.nn.softmax(self.classifier['class_2']), tf.float32)
        self.sess.run(tf.global_variables_initializer())
        return self
