# Learning with an adversary
# Author : Krishnan Raghavan
# Date: May 5th, 2018

#######################################################################################
# Define all the libraries
import random
import numpy as np
import tensorflow as tf
import operator
from functools import reduce


####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################
def sample_Z(m, n, kappa):
    return(np.random.uniform(-kappa, kappa, size=[m, n]))

def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
   high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
   return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)
############################################################################################
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
def drelu(x):
    zero = tf.zeros(x.get_shape())
    one = tf.ones(x.get_shape())
    return(tf.where(tf.greater(x, zero), one, zero))

#############################################################################################
def dtanh(x):
    return(1 - tf.multiply(tf.nn.tanh(x), tf.nn.tanh(x)))

#############################################################################################
def dsigmoid(x):
    return(tf.multiply((1 - tf.nn.sigmoid(x)), tf.nn.sigmoid(x)))

#############################################################################################
def act_ftn(name):
    if(name == "tanh"):
        return(tf.nn.tanh)
    elif(name == "relu"):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)
    else:
        print("not tanh or relu")

#############################################################################################
def dact_ftn(name):
    if(name == "tf.nn.tanh"):
        return(dtanh)
    elif(name == "tf.nn.relu"):
        return(drelu)
    elif(name == "tf.nn.sigmoid"):
        return(dsigmoid)
    else:
        print("not tanh or relu")

#############################################################################################
def init_ftn(name, num_input, num_output, runiform_range):
    if(name == "normal"):
        return(tf.truncated_normal([num_input, num_output]))
    elif(name == "uniform"):
        return(tf.random_uniform([num_input, num_output], minval=-runiform_range, maxval=runiform_range))
    else:
        print("not normal or uniform")


#############################################################################################
# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Layer_cost =[]
        self.Noise_List =[]
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()

#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act,
                 trainability, key, act_par="tf.nn.tanh"):
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

            activations = act(preactivate, name='activation' + key)
            if (act_par == "tf.nn.tanh"):
                return activations, tf.matrix_diag(dtanh(preactivate) + 0.001)
            elif(act_par == "tf.nn.relu"):
                return activations, tf.matrix_diag(drelu(preactivate) + 0.001)
            elif(act_par == "tf.nn.sigmoid"):
                return activations, tf.matrix_diag(dsigmoid(preactivate) + 0.001)

#     def assign(self):
#         train_list = []
#         for i in xrange(len(self.Layer_cost)):
#             cost, weight, bias, fan_in, batch_size, layer_in, dlayer,\
#                 backward = self.Layer_cost[i]
#             cost, weight_clamp, bias_clamp = self.Layer_cost_Clamp[i]
#             train_list.append(weight_clamp.assign(weight))
#             train_list.append(bias_clamp.assign(bias))
#         return train_list

# ##############################################################################################
    def Noise_optimizer(self, lr):
        train_list =[]
        weight, bias  =  self.Noise_List[0]
        weight_update = -1*tf.gradients(self.classifier["Overall cost"], weight)[0] + 0.0001*weight
        bias_update   = -1*tf.gradients(self.classifier["Overall cost"]  , bias)[0] + 0.0001*bias 

        ##Generate the updated variables
        train_list.append((weight_update, weight))
        train_list.append((bias_update, bias))
        return tf.train.GradientDescentOptimizer(lr).apply_gradients(train_list)

##############################################################################################
    def Custom_Optimizer(self, lr):
        train_list = []
        for i in xrange(len(self.Layer_cost)):
            weight,bias= self.Layer_cost[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall cost'], weight)[0] + 0.0001*weight
            bias_update   = tf.gradients(  self.classifier['Overall cost'], bias)[0] + 0.0001*bias
            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range = 1):
        i =1
        with tf.name_scope("Trainer_Network"):
            
            self.classifier['WeightFL_layer_11']= weight_variable([Layers[i - 1],\
             Layers[i]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11']  = bias_variable([Layers[i]],\
             trainable=False, name='BiasFL_layer_11')

            # Affine transformation 
            self.classifier['A'] = weight_variable([Layers[i-1], Layers[i-1]], trainable=False, name='A')
            self.classifier['b'] = bias_variable( [Layers[i-1]], trainable=False, name='b')

            # Noise model N = A^{T}x + b
            input_noise  = self.Deep['FL_layer_10']
            self.Deep['noisemodel'] = tf.add(tf.matmul(input_noise, self.classifier['A']), self.classifier['b'])   
            self.Noise_List.append((self.classifier['A'], self.classifier['b']))
            
            # Input data
            input_model = self.Deep['noisemodel']   
            
            # Actual First layer
            preactivate = tf.matmul(input_model, self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep['FL_layer_11'] = act_function(preactivate)
            weight_temp = self.classifier['Weight'+'FL_layer_1' + str(1)]
            bias_temp   = self.classifier['Bias'+'FL_layer_1' + str(1)]
            self.Layer_cost.append( (weight_temp, bias_temp) )     

            # Neural network for the rest of the layers.
            for i in range(2, len(Layers)):
                self.Deep['FL_layer_1' + str(i)], self.Deep['dFL_layer_1' + str(i)]\
                    = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1],
                                    Layers[i], act=act_function, trainability=False, key='FL_layer_1' + str(i))
                weight_temp = self.classifier['Weight' + 'FL_layer_1' + str(i)]
                bias_temp   = self.classifier  ['Bias'   + 'FL_layer_1' + str(i)]

                self.Layer_cost.append( (weight_temp, bias_temp) )     

        with tf.name_scope("Classifier"):
            self.classifier['class_Noise'], self.classifier['dclass_1'] = \
            self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers) - 1], classes, act=tf.identity, trainability=False, key='class')

            self.Layer_cost.append((self.classifier['Weightclass'] , self.classifier['Biasclass']))

        with tf.name_scope("Extra_Network"):
            self.Deep['FL_layer_3'+str(0)] = self.Deep['FL_layer_10'] 
            for i in range(1, len(Layers)):
                key = 'FL_layer_1' + str(i)
                preactivate = tf.matmul(self.Deep['FL_layer_3' + str(i - 1)],
                self.classifier['Weight' + key]) + self.classifier['Bias' + key]
                self.Deep['FL_layer_3'+str(i)] = act_function(preactivate, name='activation_3' + key)

            self.classifier['class_NoNoise'] = tf.identity(tf.matmul(self.Deep['FL_layer_3'+str(len(Layers)-1)],\
             self.classifier['Weightclass']) + self.classifier['Biasclass'])     

        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

##################################################### Initialize the NN ##############################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size=64,
                        back_range=1,  par='GDR', act_par="tf.nn.tanh"):
            with tf.name_scope("Placeholders"):  
                # Setup the placeholders
                self.classifier['Target']        = tf.placeholder( tf.float32, shape=[None, classes])
                self.classifier["learning_rate"] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.Deep['FL_layer_10']         = tf.placeholder(tf.float32, shape=[None, Layers[0]])

            with tf.name_scope("Network"): 
                self.Def_Network(classes, Layers, act_function, batch_size)

            with tf.name_scope("Trainer"):
                # Cross Entropy loss 
                self.classifier["cost"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.classifier['class_Noise'], labels=self.classifier['Target'], name='Error_Cost'))
                self.classifier["cost_wonoise"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.classifier['class_NoNoise'], labels=self.classifier['Target'], name='Error_Cost_wonoise'))


                self.classifier["Cost_Loss"]  = tf.reduce_mean(tf.nn.l2_loss( self.classifier['A']) \
                + tf.nn.l2_loss( self.classifier['b']))
                self.classifier["Cost_Noise"] = 0.5*(tf.reduce_mean(tf.norm( self.classifier['A'], ord = np.inf))\
                + tf.reduce_mean(tf.norm( self.classifier['b'], ord = np.inf)))

                self.classifier["Overall cost"] = 0.66*self.classifier["cost"] + (1-0.66)*self.classifier["cost_wonoise"]
                # + 0.33*self.classifier["Cost_Loss"] - self.classifier["Cost_Noise"] 

            with tf.name_scope("Optimizers"): 
                # Call the other optimizer
                self.Trainer["grad_op"]  = self.Custom_Optimizer(self.classifier["learning_rate"])
                self.Trainer["Noise_op"] = self.Noise_optimizer( self.classifier["learning_rate"])
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_NoNoise']),1 ) , 
                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())
            return self




