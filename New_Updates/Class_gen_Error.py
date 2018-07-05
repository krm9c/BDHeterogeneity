
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
            activations = act(preactivate, name='activation'+key)
        return activations, input_tensor

#############################################################################################
    def get_num_nodes_respect_batch(self,tensor_shape):
        shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0
        return reduce(operator.mul, tensor_shape.as_list()[shape_start_index:], 1), shape_start_index

#############################################################################################
    def random_matrix(self, shape):
        with tf.variable_scope("radom_matrix"):
            rand_t = tf.random_uniform(shape, -1, 1)
        return tf.Variable(rand_t, name="weights")

#############################################################################################
    def flatten_respect_batch(self,tensor):
        """
        Flattens a tensor respecting the batch dimension.
        Returns the flattened tensor and its shape as a list. Eg (tensor, shape_list).
        """
        with tf.variable_scope("flatten_respect_batch"):
            shape = tensor.get_shape()
            num_nodes, shape_start_index = self.get_num_nodes_respect_batch(shape)

            # Check if the tensor is already flat!2
            if len(shape) - shape_start_index == 1:
                return tensor, shape.as_list()

            # Flatten the tensor respecting the batch.
            if shape_start_index > 0:
                flat = tf.reshape(tensor, [-1, num_nodes])
            else:
                flat = tf.reshape(tensor, [num_nodes])

            return flat

#############################################################################################
    def reshape_respect_batch(self,tensor, out_shape_no_batch_list):
        """
        Reshapes a tensor respecting the batch dimension.
        Returns the reshaped tensor
        """
        with tf.variable_scope("reshape_respect_batch"):
            tensor_shape = tensor.get_shape()
            shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0
            # Flatten the tensor respecting the shape.
            if shape_start_index > 0:
                shaped = tf.reshape(tensor, [-1] + out_shape_no_batch_list)
            else:
                shaped = tf.reshape(tensor, out_shape_no_batch_list)
            return shaped

#############################################################################################
    def gen_feedback(self,layer_num_nodes, out_num_nodes):
        with tf.variable_scope("radom_matrix_1"):
            # s,u,v        = tf.svd(layer_grad, compute_uv=True)
            # diag_mat     = tf.sqrt( tf.abs(tf.linalg.diag(s)))
            # mod_diag_mat = tf.add(diag_mat, tf.multiply(0.1, tf.eye(tf.shape(diag_mat)[1])))
            # temp_rand    = tf.matmul(diag_mat, v, adjoint_b=True)
            # rand_t       = tf.matmul(u,temp_rand)
            Bhat = tf.random_uniform([layer_num_nodes, out_num_nodes], -1, 1)
            return Bhat
# #############################################################################################
    def error_driven_Learning(self, optimizer, loss, output, activation_param_pairs, dec):
        with tf.variable_scope("direct_feedback_alignment"):
            # Get flatten size of outputs
            out_shape = output.get_shape()
            out_num_nodes, shape_start_index = self.get_num_nodes_respect_batch(out_shape)
            # Get the loss gradients with respect to the outputs.
            loss_grad = tf.gradients(loss, output)[0]
            # loss_grad_shape = tf.Print(loss_grad, [tf.shape(loss_grad)], "Loss Grad")
            virtual_gradient_param_pairs = []
            # Construct direct feedback for each layer
            for i, (layer_out, layer_in, layer_weights) in enumerate(activation_param_pairs):
                with tf.variable_scope("virtual_feedback_{}".format(i)):
                    _, layer_shape = self.flatten_respect_batch(layer_out)
                    layer_num_nodes =layer_shape[-1]
                    # grad_list  = [tf.gradients(output[:, idx], layer_out)[0] for idx in range(out_num_nodes)]
                    # l_grad     = tf.transpose(tf.reduce_mean(tf.stack(grad_list), 1))
                    B = self.gen_feedback(layer_num_nodes, out_num_nodes)
                    Feedback = tf.matmul(loss_grad, tf.transpose(B)) 
                    su = 0
                    for weight in layer_weights:
                        if su>0:
                            Delta_w_temp = tf.reduce_mean(tf.matmul(tf.transpose(Feedback), layer_in),1)    
                        else:
                            Delta_w_temp = tf.transpose(tf.matmul(tf.transpose(Feedback), layer_in))

                        # Delta_w = tf.reshape(Delta_w_temp, tf.shape(weight))
                        virtual_gradient_param_pairs +=[(Delta_w_temp, weight)]
                        su = su+1
                        
            train_op = optimizer.apply_gradients(virtual_gradient_param_pairs)
            return train_op

#############################################################################################
    def Custom_Optimizer(self, lr):
        a = tf.gradients(self.classifier['cost_NN'], self.keys)
        b = [  (tf.placeholder("float32", shape=grad.get_shape())) for grad in a]
        var_list =[ item for item in self.keys]
        c =  tf.train.AdamOptimizer(lr).apply_gradients( [ (e,var_list[i]) for i,e in enumerate(b) ] )
        return a,b, c
        
##############################################################################################
    # def error_driven_Learning(self, optimizer, loss, output, activation_param_pairs, dec):
    #     with tf.variable_scope("direct_feedback_alignment"):
    #         # Create Lists
    #         Delta_w_list = []
    #         placeholder  = []
    #         # Get flatten size of outputs

    #         out_shape = output.get_shape()
    #         out_num_nodes, shape_start_index = self.get_num_nodes_respect_batch(out_shape)
    #         out_non_batch_shape = out_shape.as_list()[shape_start_index:]
    #         # Get the loss gradients with respect to the outputs.
    #         loss_grad = tf.gradients(loss, output)
    #         virtual_gradient_param_pairs = []
    #         flag = 0;
    #         # Construct direct feedback for each layer
    #         for i, (layer_out, layer_weights) in enumerate(activation_param_pairs):
    #             with tf.variable_scope("virtual_feedback_{}".format(i)):
    #                 if layer_out is output:
    #                     proj_out = output
    #                     flag = 1
    #                 else:
    #                     # Calculate the feedback first
    #                     flat_layer, layer_shape = self.flatten_respect_batch(layer_out)

    #                     print("Shape of the layer is", layer_shape)
    #                     layer_num_nodes = layer_shape[-1]
    #                     grad_list  = [tf.gradients(output[:, idx], layer_out)[0] for idx in range(out_num_nodes)]
    #                     l_grad     = tf.transpose(tf.reduce_mean(tf.stack(grad_list), 1))
    #                     B = self.gen_feedback(l_grad, layer_num_nodes, out_num_nodes)
    #                     flat_proj_out = tf.matmul(flat_layer, B)

    #                     ## Calculate the decay components
    #                     layer_out = layer_out + tf.random_normal(tf.shape(layer_out), mean= 0.0, stddev = 0.001)
    #                     s,_,_    = tf.svd(layer_out)
    #                     diag_mat = tf.sqrt( tf.abs(tf.linalg.diag(s))+0.001);
    #                     sum_diag = (tf.reduce_sum(diag_mat)+0.01)
    #                     div_diag_mat = tf.truediv(diag_mat, sum_diag, name=None)

    #                     # Reshape back to output dimensions and then get the gradients.
    #                     proj_out  = self.reshape_respect_batch(flat_proj_out, out_non_batch_shape)
    #                     fac = tf.subtract(tf.eye(tf.shape(diag_mat)[1]),  0.001*div_diag_mat)

    #                 j = 0;
    #                 for weight in layer_weights:
    #                     if flag == 0:
    #                         if j>0:
    #                             reg  = dec*tf.squeeze( tf.matmul( fac  ,tf.expand_dims(weight,1) ), axis = 1)
    #                         else:
    #                             reg  = dec*tf.matmul(weight,fac)
    #                     else:
    #                         reg = dec*weight
    #                     j= j+1;
    #                     Delta_w = tf.gradients( proj_out , weight, grad_ys=loss_grad)[0]

    #                     virtual_gradient_param_pairs +=  [
    #                        ( tf.div(Delta_w,1+tf.linalg.norm(Delta_w)) + reg, weight)]
    #         train_op = optimizer.apply_gradients(virtual_gradient_param_pairs)
    #         return train_op

#############################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, par='GDR'):
        if par =='EDL':
            array = []
        with tf.name_scope("FLearners_1"):
            self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])
            for i in range(1,len(Layers)):
                self.Deep['FL_layer'+str(i)], self.Deep['FL_layer'+str(i)+'preactivate'] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1],\
                Layers[i], act= act_function, trainability = False, key = 'FL_layer'+str(i))
                self.keys.append(self.classifier['Weight'+'FL_layer'+str(i)])
                self.keys.append(self.classifier['Bias'+'FL_layer'+str(i)])
                if par == 'EDL':
                    array.append(\
                        (self.Deep['FL_layer' + str(i)], self.Deep['FL_layer' + str(i) + 'preactivate'],
                    ([self.classifier['Weight'+'FL_layer'+str(i)],self.classifier['Bias'+'FL_layer'+str(i)] ]) ) )

        with tf.name_scope("Classifier"):
            self.classifier['class'], self.classifier['class_preactivate'] = self.nn_layer( self.Deep['FL_layer'+str(len(Layers)-1)],\
                                                                                            Layers[len(Layers) - 1], classes, act=tf.identity,  trainability=False, key='class')
            self.keys.append(self.classifier['Weightclass'])
            self.keys.append(self.classifier['Biasclass'])
            if par == 'EDL':
                    array.append(\
                    (self.classifier['class'], self.classifier['class_preactivate'], (self.classifier['Weightclass'],self.classifier['Biasclass']) ) )

        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

        if par  is not 'EDL':
            with tf.name_scope("Trainer"):
                global_step = tf.Variable(0, trainable=False)
                Error_Loss =  tf.nn.softmax_cross_entropy_with_logits(logits = \
                self.classifier['class'], labels = self.classifier['Target'], name='Cost')
                Reg = 0.01
                for element in self.keys:
                    Reg = Reg+tf.nn.l2_loss(element)
                # The final cost function
                self.classifier["cost_NN"] = tf.reduce_mean(Error_Loss)
                self.Trainer["grads"], self.Trainer["grad_placeholder"], self.Trainer["apply_placeholder_op"] =\
                self.Custom_Optimizer(lr)

                tf.summary.scalar('LearningRate', lr)
                tf.summary.scalar('Cost_NN', self.classifier["cost_NN"])
        else:
            with tf.name_scope("Trainer"):
                global_step = tf.Variable(0, trainable=False)
                dec = 0.00001 # tf.train.exponential_decay(lr, global_step,
                              #              100000, 0.99, staircase=True)
                Error_Loss =  tf.nn.softmax_cross_entropy_with_logits(logits = \
                self.classifier['class'], labels = self.classifier['Target'], name='Cost'
                )
                # The final cost function
                self.classifier["cost_NN"] = tf.reduce_mean(Error_Loss)   
                self.Trainer["EDL"]= self.error_driven_Learning(
                    tf.train.GradientDescentOptimizer(dec),
                    Error_Loss, self.classifier['class'], array, dec)

        with tf.name_scope('Evaluation'):
            with tf.name_scope('CorrectPrediction'):
                self.Evaluation['correct_prediction'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']) ,1),\
                tf.argmax(self.classifier['Target'],1))

            with tf.name_scope('Accuracy'):
                self.Evaluation['accuracy'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction'], tf.float32))

            with tf.name_scope('Prob'):
                self.Evaluation['prob'] = tf.cast( tf.nn.softmax(self.classifier['class']), tf.float32 )
        self.sess.run(tf.global_variables_initializer())

        return self
