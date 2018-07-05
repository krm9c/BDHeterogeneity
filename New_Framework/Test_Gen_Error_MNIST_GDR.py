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
def return_dict(placeholder, List, model, batch_x, batch_y, lr):
    S = {}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer_10']] = batch_x
    S[model.classifier['Target']] = batch_y
    S[model.classifier["learning_rate"]] = lr 
    return S


def sample_Z(X, m, n, kappa):
    return (X+np.random.uniform(-kappa, kappa, size=[m, n]))
    #return (X + np.random.normal(0, kappa, size=[m, n]))
####################################################################################
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(4)]
    depth.extend(L)
    lr = 0.001
    model = model.init_NN_custom(classes, lr, depth, tf.nn.relu)
    acc_array = np.zeros((Train_Glob_Iterations, 1))
    acc_array_train = np.zeros((Train_Glob_Iterations, 1))
    noise_cost = np.zeros((Train_Glob_Iterations, 1))
    cost_error = np.zeros((Train_Glob_Iterations, 1))
    Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=4)
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        for i in tqdm(t):
            lr = 0.99*lr
            for j in xrange(Train_noise_Iterations):
                for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                    batch_xs, batch_ys = batch
                    # Update the noise
                    _ = model.sess.run([model.Trainer["apply_noise_op"]],\
                    feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']: batch_ys, model.classifier["learning_rate"]:lr})
                    break

            for j in xrange(10*Train_noise_Iterations):
                for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                    batch_xs, batch_ys = batch
                    # Gather Gradients
                    grads_1 = model.sess.run([model.Trainer["grads"]],
                    feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']: batch_ys, model.classifier["learning_rate"]:lr})
                    List = [g for g in grads_1[0]]
                    _ = model.sess.run([ model.Trainer["apply_placeholder_op"]],
                    feed_dict=return_dict(model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys, lr))

            if i % 1 == 0:  
                # Evaluation and display part 
                acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']:Noise_data, model.classifier['Target']: y_test, model.classifier["learning_rate"]:lr})
                acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train, model.classifier["learning_rate"]:lr})

                # The individual costs
                noise_cost[i] = model.sess.run([model.classifier["Noise_Loss"]],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train, model.classifier["learning_rate"]:lr})
                cost_error[i] = model.sess.run([model.classifier["Error_Loss"]],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train,model.classifier["learning_rate"]:lr})
                # Print all the outputs
                print("Accuracy", i, "Test", acc_array[i],"Train", acc_array_train[i])
                print("Cost Error", cost_error[i], "Noise cost", noise_cost[i])

                # Stop the learning in case of this condition being satisfied
                if max(acc_array) > 0.99:
                    break

        acc_array = np.zeros((len(kappa),1))
        acc_array_train = np.zeros((len(kappa), 1))

        for i,element in enumerate(kappa):
            Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=element)
            acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
            feed_dict={model.Deep['FL_layer_10']:Noise_data, model.classifier['Target']: y_test})
            acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
            feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train})

    except Exception as e:
        print("I found an exception", e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    gc.collect()
    return np.reshape(acc_array, (len(kappa))), np.reshape(acc_array_train, (len(kappa))), np.reshape( (acc_array-acc_array_train), (len(kappa)))


# Setup the parameters and call the functions
Train_batch_size =64
Train_Glob_Iterations = 100
Train_noise_Iterations = 1
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
filename = 'mnist_grad_gen_gaus_w_noise_l2_los.csv'
iterat_kappa = 100
Kappa_s = np.random.uniform(0, 4, size=[iterat_kappa])
Results = np.zeros([iterat_kappa, 4])
print("Details", filename, "MNIST Grad -- New Framework", iterat_kappa,
      Train_batch_size, Train_Glob_Iterations)
Results[:, 0] = Kappa_s
Results[:, 1], Results[:, 2], Results[:, 3] = Analyse_custom_Optimizer_GDR_old(
X_train, y_train, X_test, y_test, Kappa_s)
np.savetxt(filename, Results, delimiter=',')