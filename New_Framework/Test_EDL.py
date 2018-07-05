# The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import Class_Gen_Van as NN_class
import tensorflow as tf
import numpy as np
import traceback
###################################################################################
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

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):
    S = {}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer_10']] = batch_x
    S[model.classifier['Target']] = batch_y
    return S
def sample_Z(X, m, n, kappa):
    return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    #return (X + np.random.normal(0, kappa, size=[m, n]))

####################################################################################
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, kappa):
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
    noise_cost = np.zeros((Train_Glob_Iterations, 1))
    cost_error = np.zeros((Train_Glob_Iterations, 1))
    cost_gen   = np.zeros((Train_Glob_Iterations, 1))
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        Noise_data = sample_Z(
            X_test, X_test.shape[0], X_test.shape[1], kappa=kappa)
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                # Gather Gradients
                grads_1 = model.sess.run([model.Trainer["grads"]],
                feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']: batch_ys})
                List = [g for g in grads_1[0]]
                _ = model.sess.run([ model.Trainer["apply_placeholder_op"]],
                feed_dict=return_dict(model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys))

            if i % 1 == 0:  
                if i%(Train_noise_Iterations*2) == 0:
                    for j in xrange(Train_noise_Iterations):
                        for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                            batch_xs, batch_ys = batch
                            # Update the noise
                            _ = model.sess.run([model.Trainer["apply_noise_op"]],
                            feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']: batch_ys})

                # Evaluation and display part 
                acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']:Noise_data, model.classifier['Target']: y_test})
                acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train})
                noise_output = model.sess.run([model.Evaluation['accuracy_1']],
                feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']: y_test})

                # The individual costs
                noise_cost[i] = model.sess.run([model.classifier["Noise_Loss"]],\
                feed_dict={model.Deep['FL_layer_10']: Noise_data, model.classifier['Target']: y_test})
                cost_error[i] = model.sess.run([model.classifier["Error_Loss"]],\
                feed_dict={model.Deep['FL_layer_10']: Noise_data, model.classifier['Target']: y_test})
                cost_gen[i] = model.sess.run([model.classifier["Gen_Loss"]],
                feed_dict={model.Deep['FL_layer_10']: Noise_data, model.classifier['Target']: y_test})
                
                
                # Print all the outputs
                print("Accuracy", i, "Test", acc_array[i], "Noise", noise_output, "Train", acc_array_train[i], "gen_error", (acc_array[i] - acc_array_train[i]))
                print("Cost Error", cost_error[i], "Noise cost", noise_cost[i], "gen cost", cost_gen[i])

                # Stop the learning in case of this condition being satisfied
                if max(acc_array) > 0.99:
                    pr = model.sess.run([model.Evaluation['prob']], feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']: y_test})
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


# Setup the parameters and call the functions
Train_batch_size = 64
Train_Glob_Iterations = 1000
Train_noise_Iterations = 5
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
classes = 10
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

print("Train", X_train.shape, "Test", X_test.shape)
inputs = X_train.shape[1]
filename = 'mnist_grad_gen_gaus_Ty.csv'
iterat_kappa = 1
Kappa_s = [0.5] # np.random.uniform(0, 1, size=[iterat_kappa])
Results = np.zeros([iterat_kappa, 4])

print("Details", filename, "MNIST Grad -- New Framework", iterat_kappa,
      Train_batch_size, Train_Glob_Iterations)

for i in tqdm(xrange(iterat_kappa)):
    print("i", i, "kappa", Kappa_s[i])
    Results[i, 0] = Kappa_s[i]
    Results[i, 1], Results[i, 2], Results[i, 3] = Analyse_custom_Optimizer_GDR_old(
    X_train, y_train, X_test, y_test, Kappa_s[i])
np.savetxt(filename, Results, delimiter=',')
