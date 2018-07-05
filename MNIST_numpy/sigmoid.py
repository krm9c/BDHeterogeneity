import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#  tanh
def tanh(z):
    """The tanh"""
    return np.tanh(z)

def tanh_prime(z):
    """Derivative of the tanh function."""
    return (1-tanh(z)*tanh(z))

#  relu
def relu(z):
    """The relu function."""
    return x*(x > 0)

def relu_prime(z):
    """Derivative of the relu"""
    return 1*(x > 0)
