# The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import Class_gen_Error as  NN_class
import tensorflow as tf
import numpy as np
import traceback

################################################################################
def import_pickled_data(string):
    f = gzip.open('../../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

################################################################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):
    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S

################################################################################
def return_dict_EDL(model, batch_x, batch_y):
    S ={}
    S[model.Deep['FL_layer0']] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S
################################################################################
def sample_Z(X, m, n, kappa):
    return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    # return (X + np.random.normal(0, kappa, size=[m, n]))
#####################################################################################
def Analyse_GDR_with_Hetero(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(3)]
    depth.extend(L)
    model = model.init_NN_custom( classes, 0.01, depth, tf.nn.relu)
    acc_array = np.zeros((Train_Glob_Iterations, 1))
    acc_array_train = np.zeros((Train_Glob_Iterations, 1))
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        Noise_data = sample_Z(
            X_test, X_test.shape[0], X_test.shape[1], kappa=kappa)

        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                batch_noise_xs = sample_Z(
                    batch_xs, Train_batch_size, X_train.shape[1], 1)
                grads_1 = model.sess.run([model.Trainer["grads"]],
                                         feed_dict={model.Deep['FL_layer0']: batch_xs, model.classifier['Target']: batch_ys})
                grads_2 = model.sess.run([model.Trainer["grads"]],
                                         feed_dict={model.Deep['FL_layer0']: batch_noise_xs, model.classifier['Target']: batch_ys})

                List_1 = [g for g in grads_1[0]]
                List_2 = [g for g in grads_2[0]]
                List   = [np.add(a, b) for a, b in zip(List_1, List_2)]
                # Apply gradients
                model.sess.run([model.Trainer["apply_placeholder_op"]],
                                            feed_dict=return_dict(model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys))

            if i % 1 == 0:
                acc_array[i] = model.sess.run([model.Evaluation['accuracy']],
                                                       feed_dict={model.Deep['FL_layer0']: Noise_data, model.classifier['Target']: y_test})
                acc_array_train[i] = model.sess.run([model.Evaluation['accuracy']],
                                                             feed_dict={model.Deep['FL_layer0']: X_train, model.classifier['Target']: y_train})
                print("iteration Accuracy", acc_array[i],  "Train", acc_array_train[i], "gen_error", (
                    acc_array[i]-acc_array_train[i]))
    except Exception as e:
        print("I found an exception", e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    gc.collect()
    print("iteration Accuracy -- Test", acc_array[i],  "Train", acc_array_train[i], "gen_error", (acc_array[i]-acc_array_train[i]))
    return acc_array[i], acc_array_train[i], (acc_array[i]-acc_array_train[i])


####################################################################################
def Analyse_GDR_without_Hetero(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(3)]
    depth.extend(L)
    model = model.init_NN_custom(
            classes, 0.01,  depth, tf.nn.relu)
    acc_array = np.zeros((Train_Glob_Iterations, 1))
    acc_array_train = np.zeros((Train_Glob_Iterations, 1))
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        Noise_data = sample_Z(  X_test, X_test.shape[0], X_test.shape[1], kappa=kappa)
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                # Gather Gradients
                grads_1 = model.sess.run([model.Trainer["grads"]],
                                         feed_dict={model.Deep['FL_layer0']: batch_xs, model.classifier['Target']: batch_ys})
                List = [g for g in grads_1[0]]
                model.sess.run([ model.Trainer["apply_placeholder_op"]],
                                            feed_dict=return_dict(model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys))
            if i % 1 == 0:
                acc_array[i] = model.sess.run([model.Evaluation['accuracy']],
                                                       feed_dict={model.Deep['FL_layer0']: Noise_data, model.classifier['Target']: y_test})
                acc_array_train[i] = model.sess.run([model.Evaluation['accuracy']],
                                                             feed_dict={model.Deep['FL_layer0']: X_train, model.classifier['Target']: y_train})
                print("Accuracy", acc_array[i],  acc_array_train[i],
                      "gen_error", (acc_array[i]-acc_array_train[i]))

                if max(acc_array) > 0.99:
                    pr = model.sess.run([model.Evaluation['prob']],
                                                 feed_dict={model.Deep['FL_layer0']: Noise_data, model.classifier['Target']: y_test})
                    break
    except Exception as e:
        print("I found an exception", e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    gc.collect()
    print("Accuracy", acc_array[i],  acc_array_train[i],
          "gen_error", (acc_array[i]-acc_array_train[i]))
    return acc_array[i], acc_array_train[i], (acc_array[i]-acc_array_train[i])


#####################################################################################
def Analyse_EDL(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(3)]
    depth.extend(L)

    model = model.init_NN_custom(classes, 0.001, depth, tf.nn.relu,'EDL')
    acc_array = np.zeros( (Train_Glob_Iterations , 1) )
    acc_array_train = np.zeros((Train_Glob_Iterations , 1))

    try:
        t = xrange(Train_Glob_Iterations)

        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys  = batch
                batch_xs_noise = sample_Z(batch_xs, batch_xs.shape[0], batch_xs.shape[1], 1)
                model.sess.run( [model.Trainer["EDL"]] , feed_dict= return_dict_EDL(model, batch_xs, batch_ys) )
                model.sess.run( [model.Trainer["EDL"]] , feed_dict= return_dict_EDL(model, batch_xs_noise, batch_ys) )

            if i % 1 == 0:
                X =sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa)
                acc_array[i]  = model.sess.run( [ model.Evaluation['accuracy'] ],\
                feed_dict = {model.Deep['FL_layer0']: X, model.classifier['Target']: y_test})

                acc_array_train[i]  = model.sess.run( [ model.Evaluation['accuracy'] ],\
                feed_dict = {model.Deep['FL_layer0']: X_train, model.classifier['Target']: y_train})
                print "Accuracy", "test", acc_array[i],  "train", acc_array_train[i], "gen_error", (acc_array[i]-acc_array_train[i])

        tf.reset_default_graph()
        del model
        gc.collect()

    except Exception as e:
        print(e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    gc.collect()
    print("Accuracy", acc_array[i],  acc_array_train[i], "gen_error", (acc_array[i]-acc_array_train[i]))
    return acc_array[i], acc_array_train[i],(acc_array[i]-acc_array_train[i])

## Setup the parameters and call the functions
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

#####################################################################################
Train_batch_size = 128
Train_Glob_Iterations = 50
classes = 10
dataset = 'mnist'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test  = mnist.test.images
y_train = mnist.train.labels
y_test  = mnist.test.labels
inputs   = X_train.shape[1]
classes  = y_train.shape[1]
filename = "mnist_EDL.csv"
iterat_kappa = 1
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
Results = np.zeros([iterat_kappa,7])
#####################################################################################
### Final Loop
print("Four Layers", "Details", filename, dataset, iterat_kappa, Train_batch_size, Train_Glob_Iterations)
for i in tqdm(xrange(iterat_kappa)):
    print("i", i, "kappa", Kappa_s[i])
    Results[i,0] = Kappa_s[i]
    print("EDL")
    ## EDL
    Analyse_EDL(X_train,y_train,X_test,y_test, Kappa_s[i])
    print("GDR Hetero")
    ## GDR Prop Framework
    Analyse_GDR_without_Hetero(X_train, y_train, X_test, y_test, Kappa_s[i])
    print("GDR without Hetero")
    # GDR without Framework
    Analyse_GDR_with_Hetero(X_train, y_train, X_test, y_test, Kappa_s[i])
