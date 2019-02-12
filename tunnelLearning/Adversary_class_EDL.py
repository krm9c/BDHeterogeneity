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
#############################################################################################
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


##############################################################################################
    def Noise_optimizer(self, lr):
        train_list =[]
        cost, weight, bias, fan_in, batch_size, layer_in, dlayer,backward = self.Layer_cost[0]
        weight_update = tf.gradients(self.classifier["Overall cost"], weight)[0] + 0.0001*weight
        # #Generate the updated variables
        train_list.append((weight_update, weight))
        return tf.train.GradientDescentOptimizer(lr).apply_gradients(train_list)

##################################################################################
    def Optimizer(self, lr, dError_dy):
        train_list = []
        a = []
        var_list = []
        self.classifier['real_cost'] = []
        self.classifier['updates'] = []
        for i in xrange(len(self.Layer_cost)):
            cost, weight, bias, fan_in, batch_size, layer_in, dlayer,\
                backward = self.Layer_cost[i]
            
            # cost_clamp, weight_clamp, bias_clamp = self.Layer_cost_Clamp[i]
            # self.classifier['real_cost'].append(    
            #     tf.nn.l2_loss(cost - cost_clamp))
            
            if i < (len(self.Layer_cost) - 1):
                temp_mul = tf.matmul(backward, dlayer)
                # temp_mul = backward
                dError_dhidden = tf.matmul(dError_dy, temp_mul)
                reshaped_layer_in = tf.reshape(
                    layer_in, [batch_size, fan_in, 1])
                upd_weight = tf.reduce_mean(
                    tf.matmul(reshaped_layer_in, dError_dhidden), 0)
                upd_bias = tf.reduce_mean(dError_dhidden, 0)

                # #### Gradient Descent update
                # dir_weight_update = ((tf.gradients(self.classifier['real_cost'][i], weight)[0])
                #                      - (tf.gradients(self.classifier['real_cost'][i], weight_clamp)[0]))
                # self.classifier['updates'].append(
                #     tf.nn.l2_loss(dir_weight_update))
                # dir_weight_update = tf.truediv( dir_weight_update, (tf.norm(dir_weight_update) + 0.001))
                # dir_bias_update = ((tf.gradients(self.classifier['real_cost'][i], bias)[0])\
                # - (tf.gradients(self.classifier['real_cost'][i], bias_clamp)[0]))
                # dir_bias_update = tf.truediv(dir_bias_update, (tf.norm(dir_bias_update) + 0.001))
                # weight_update = tf.add((upd_weight),  0.01*dir_weight_update) + 0.0001*weight
                # bias_update   = tf.add((upd_bias[0]), 0.01*dir_bias_update)   + 0.0001*bias

                weight_update = upd_weight  + 0.0001*weight
                bias_update   = upd_bias[0] + 0.0001*bias
            else:
                reshaped_layer_in = tf.reshape(
                    layer_in, [batch_size, fan_in, 1])

                dError_dhidden = dError_dy
                upd_weight = tf.reduce_mean(
                    tf.matmul(reshaped_layer_in, dError_dhidden), 0)

                upd_bias = tf.reduce_mean(dError_dy, 0)

                weight_update = upd_weight   + 0.0001*weight
                bias_update  = upd_bias[0]   + 0.0001*bias
                
           # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)

##############################################################################################
    def Custom_Optimizer(self, lr):
        train_list = []
        for i in xrange(len(self.Layer_cost)):
            cost, weight, bias, fan_in, \
            batch_size, layer_in, dlayer, backward = self.Layer_cost[i]

            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall cost'], weight)[0] + 0.001*weight
            bias_update   = tf.gradients(self.classifier['Overall cost'], bias)[0]   + 0.001*bias

            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range = 1):
        i =1
        with tf.name_scope("Trainer_Network"):
            
            self.classifier['WeightFL_layer_11'] = weight_variable(
            [Layers[i - 1], Layers[i]], trainable=False, name='WeightFL_layer_11')

            self.classifier['BiasFL_layer_11'] = bias_variable(
                [Layers[i]], trainable=False, name='BiasFL_layer_11')

            # Noise model N = f(W^TW(x) + b)
            input_noise = tf.Variable(tf.random_uniform([1, Layers[0]], minval= -1, maxval= 1, dtype=tf.float32))
            
            # input_noise = self.Deep['FL_layer_10']
            self.Deep['noisemodel'] = tf.matmul( tf.nn.sigmoid(input_noise) ,
                                      tf.matmul(self.classifier['WeightFL_layer_11'],\
                                      tf.transpose(self.classifier['WeightFL_layer_11']) ))  

            # Input data
            input_model   = self.Deep['FL_layer_10'] + self.Deep['noisemodel']
            # Actual First layer
            preactivate = tf.matmul(
                input_model, self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep['FL_layer_11'] = act_function(preactivate)
            # The first layer derivative
            self.Deep['dFL_layer_11'] = tf.matrix_diag(dtanh(preactivate) + 0.001)

            # Neural network for the rest of the layers.
            for i in range(2, len(Layers)):
                self.Deep['FL_layer_1' + str(i)], self.Deep['dFL_layer_1' + str(i)]\
                    = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1],
                                    Layers[i], act=act_function, trainability=False, key='FL_layer_1' + str(i))
                fan_in = Layers[i - 2]
                fan_out = Layers[i - 1]
                num_final = classes
                cost_layer = self.Deep['FL_layer_1' + str(i)]
                weight_temp = self.classifier['Weight' +
                                                'FL_layer_1' + str(i - 1)]
                
                bias_temp = self.classifier['Bias' + 'FL_layer_1' + str(i - 1)]
                layer_in = self.Deep['FL_layer_1' + str(i - 2)]
                dlayer = self.Deep['dFL_layer_1' + str(i - 1)]

                backward_t = tf.Variable(
                    init_ftn("uniform", num_final, fan_out, back_range))
                backward = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                    batch_size, num_final, fan_out])

                
                self.Layer_cost.append(
                    (cost_layer, weight_temp, bias_temp,
                        fan_in, batch_size, layer_in, dlayer, backward))

                self.Noise_List.append(
                    (cost_layer, weight_temp, bias_temp,
                     fan_in, batch_size, layer_in, dlayer, backward))
##################################################### Layer d-1 #######################################################################
#################### Layer wise information for generating the updates
        with tf.name_scope("Classifier"):
            self.classifier['class_Noise'], self.classifier['dclass_1']=self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
##################################################### Layer d-1 #######################################################################
#################### Layer wise information for generating the updates
            fan_in      =   Layers[len(Layers) - 2]
            fan_out     =   Layers[len(Layers) - 1]
            num_final   =   classes
            cost_layer  =   self.classifier['class_Noise']
            weight_temp =   self.classifier['Weight' + 'FL_layer_1' + str(len(Layers) - 1)]
            bias_temp   =   self.classifier['Bias'   + 'FL_layer_1' + str(len(Layers) - 1)]
            layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 2)]
            dlayer      =   self.Deep['dFL_layer_1' + str(len(Layers) - 1)]
            backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
            backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                            batch_size, num_final, fan_out ])
            self.Layer_cost.append((cost_layer, weight_temp, bias_temp,
            fan_in, batch_size, layer_in, dlayer, backward))

        ############ The network without the noise, primarily for output estimation
        self.Deep['FL_layer_3' + str(0)] = self.Deep['FL_layer_10']
        for i in range(1, len(Layers)):
            key = 'FL_layer_1' + str(i)
            preactivate = tf.matmul(self.Deep['FL_layer_3' + str(i - 1)],
                                    self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            self.Deep['FL_layer_3' +str(i)] \
            = act_function(preactivate, name='activation_3' + key)

        self.classifier['class_NoNoise'] = tf.identity(tf.matmul(self.Deep['FL_layer_3' + str(
            len(Layers) - 1)], self.classifier['Weightclass']) + self.classifier['Biasclass'])


    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size=64,
                        back_range=1,  par='GDR', act_par="tf.nn.tanh"):
            i = 1
            num_final = classes
        ########################### Initial Definitions
            with tf.name_scope("FLearners_1"):  

                # Setup the placeholders
                self.classifier['Target'] = tf.placeholder(
                    tf.float32, shape=[None, classes])
                
                self.classifier["learning_rate"] = tf.placeholder(
                    tf.float32, [], name='learning_rate')
                
                self.Deep['FL_layer_10'] = tf.placeholder(
                    tf.float32, shape=[None, Layers[0]])
        
        ########################################################## The network
                self.Def_Network(classes, Layers, act_function, batch_size)
        
        ################################################# Design the trainer for the  network
            with tf.name_scope("Trainer"):
                # Cross Entropy loss 
                self.classifier["cost_NN"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class_Noise'], labels=self.classifier['Target'], name='Error_Cost'))
                self.classifier["cost_NN_nonoise"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class_NoNoise'], labels=self.classifier['Target'], name='Error_Cost'))

                fan_in      =   Layers[len(Layers) - 1]
                fan_out     =   classes
                num_final   =   classes
                cost_layer  =   self.classifier["cost_NN"]
                weight_temp =   self.classifier['Weightclass']
                bias_temp   =   self.classifier['Biasclass']
                layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 1)]
                backward_t  =   tf.Variable(tf.eye(classes))
                dlayer      =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                batch_size, num_final, num_final ])
                backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
                backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                batch_size, num_final, fan_out ])

                self.Layer_cost.append(
                (cost_layer, weight_temp, bias_temp,
                fan_in, batch_size, layer_in, dlayer, backward))
                # Noise loss 
                # L2 Loss
                self.classifier["Noise_cost"] = tf.reduce_mean(tf.norm(self.Deep['noisemodel'], ord = np.inf))
                # Los Loss
                self.classifier["Cost_Noise"] = tf.reduce_mean(tf.nn.log_softmax(logits = self.Deep['noisemodel']))

        ##################################################### The cost for the unclamped network ##############################################      
                # The final cost
                self.classifier["Overall cost"] = 0.01*self.classifier["cost_NN"]\
                +(1-0.01)*self.classifier["cost_NN_nonoise"]-self.classifier["Cost_Noise"]

        ##########################################################################################################
        # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer["apply_placeholder_op"] = self.Custom_Optimizer(self.classifier["learning_rate"])
                self.Trainer["Noise_op"] = self.Noise_optimizer(self.classifier["learning_rate"])

                # The general optimizer
                dError_dy = tf.add(tf.reshape(tf.gradients(
                self.classifier["Overall cost"], self.classifier['class_Noise'])[0], [batch_size, 1, num_final])
                , tf.reshape(tf.gradients(
                self.classifier["Overall cost"], self.classifier['class_Noise'])[0], [batch_size, 1, num_final]))
                # the optimizers
                self.Trainer['optimization_op'] = self.Optimizer(
                self.classifier["learning_rate"],  dError_dy)

        ############## The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_NoNoise']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())
            return self




