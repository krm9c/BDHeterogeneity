
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
# Setup up parallelization schemes
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
####################################################################################
def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1*np.sqrt(4.0/(fan_in + fan_out))
   high = 1*np.sqrt(4.0/(fan_in + fan_out))
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
    with tf.name_scope('summaries'+key):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'+key):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#############################################################################################
# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.Summaries = {}
        self.keys = []
        self.preactivations = []
        self.sess = tf.Session(config = config)
#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights'+key):
                self.classifier['Weight'+key] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+key)
            with tf.name_scope('bias'+key):
                self.classifier['Bias'+key] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+key)
            with tf.name_scope('Wx_plus_b'+key):
                preactivate = tf.matmul(input_tensor, self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                self.preactivations.append(preactivate)
            activations = act(preactivate, name='activation'+key)
        return activations
#############################################################################################
    def Custom_Optimizer(self, lr):
        a = tf.gradients(self.classifier['cost_NN'], self.keys)
        b = [  (tf.placeholder("float32", shape=grad.get_shape())) for grad in a]
        var_list =[ item for item in self.keys]
        c =  tf.train.AdamOptimizer(lr).apply_gradients( [ (e,var_list[i]) for i,e in enumerate(b) ] )
        return a,b,c

#############################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, par='GDR'):
        with tf.name_scope("FLearners_1"):
            self.Deep['FL_layer_10'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])
            for i in range(1,len(Layers)):
            
                self.Deep['FL_layer_1'+str(i)] = self.nn_layer(self.Deep['FL_layer_1'+str(i-1)], Layers[i-1],\
                Layers[i], act= act_function, trainability = False, key = 'FL_layer_1'+str(i))
                # Put all the keys in an array
                self.keys.append(self.classifier['Weight'+'FL_layer_1'+str(i)])
                self.keys.append(self.classifier['Bias'+'FL_layer_1'+str(i)])

        with tf.name_scope("FLearners_2"):
            # initial  =   tf.random_uniform( [1,Layers[0]], minval=-1, maxval= 1, dtype=tf.float32)
            initial  =    tf.random_normal([1,Layers[0]], mean = 0.0, stddev=1)
            self.Deep['noise']= tf.Variable(initial, trainable=False, name='noise')
            self.Deep['FL_layer_2'+str(0)] = tf.add(self.Deep['FL_layer_10'], tf.reduce_mean(self.Deep['noise'], axis = 0) )
  
            for i in range(1,len(Layers)):
                key = 'FL_layer_1'+str(i)
                preactivate = tf.matmul(self.Deep['FL_layer_2'+str(i-1)], self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                self.Deep['FL_layer_2'+str(i)] = act_function(preactivate, name='activation_2'+key)

        with tf.name_scope("Classifier"):
            class_act = tf.identity
            self.classifier['class_1'] = self.nn_layer(self.Deep['FL_layer_1'+str(len(Layers)-1)],\
            Layers[len(Layers)-1], classes, act=class_act, trainability =  False, key = 'class')
            self.keys.append(self.classifier['Weightclass'])
            self.keys.append(self.classifier['Biasclass'])            
            self.classifier['class_2'] = class_act(tf.matmul(self.Deep['FL_layer_2' + str(
                len(Layers) - 1)], self.classifier['Weightclass']) + self.classifier['Biasclass'])

        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

        with tf.name_scope("Trainer"):
            # Error
            self.classifier["Error_Loss"] =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= self.classifier['class_1'], labels = self.classifier['Target'], name='Error_Cost') )
            
            # approximated Generalization error
            self.classifier["Gen_Loss"]   =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= self.classifier['class_2'], labels = self.classifier['Target'], name='Gen_Error_Cost') )
            
            # Perturbation
            self.classifier["Noise_Loss"] =  0.01*(-tf.log(tf.nn.softmax(self.Deep['noise'])))
           
            # Total Loss
            self.classifier["cost_NN"] = (self.classifier["Error_Loss"] + self.classifier["Gen_Loss"]) \
            - self.classifier["Noise_Loss"]
            
            self.Trainer["grads"], self.Trainer["grad_placeholder"], self.Trainer["apply_placeholder_op"] =\
            self.Custom_Optimizer(lr)
            
            # Train Noise
            self.Trainer["noise_grad"] = tf.gradients(
                self.classifier['cost_NN'], self.Deep['noise'])
            b = [grad for grad in self.Trainer["noise_grad"]]
            var_list =  [self.Deep["noise"]]
            self.Trainer["apply_noise_op"] = \
            tf.train.AdamOptimizer(lr).apply_gradients(\
                [ (e,var_list[i]) for i,e in enumerate(b) ])

        with tf.name_scope('Evaluation'):
            with tf.name_scope('CorrectPrediction'):
                self.Evaluation['correct_prediction'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_1']) ,1),\
                tf.argmax(self.classifier['Target'],1))
                self.Evaluation['correct_prediction_1'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_2']) ,1),\
                tf.argmax(self.classifier['Target'],1))
            with tf.name_scope('Accuracy'):
                self.Evaluation['accuracy'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction'], tf.float32))
                self.Evaluation['accuracy_1'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction_1'], tf.float32))
            with tf.name_scope('Prob'):
                self.Evaluation['prob'] = tf.cast(tf.nn.softmax(self.classifier['class_1']), tf.float32)

        self.sess.run(tf.global_variables_initializer())

        return self
