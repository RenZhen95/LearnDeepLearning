'''
A simple implementation of a feed-forward neural network, implemented
based on Michael Nielsen's book and code

``Neural Networks and Deep Learning``

This code is one of the homework assigned to implement a fully 
matrix-based approach to backpropagation over a mini-batch instead
of looping over each sample in the mini-batch, and calculating
the corresponding gradients of the cost function w.r.t the weights
and biases.
'''

# Standard library
import random
import time

# Third-party libraries
import sys
import numpy as np

# Sigmoid function
sigmoid = lambda z : 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid function
sigmoid_prime = lambda z : sigmoid(z)*(1-sigmoid(z))

class Network_MatrixBased(object):

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.

        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.

        Note that the first layer is assumed to be an input layer, and
        by convention no biases will be set for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Biases for each unit
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Weights between each unit in the next layer and each unit
        # in the previous layer
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is INPUT."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(
            self, training_data, epochs, mini_batch_size, lr,
            test_data=None
    ):
        """
        Train the neural network using mini-batch stochastic gradient
        descent.

        The ``training_data`` is a list of tuples ``(x, y)`` representing
        the training inputs and the desired outputs.

        The other non-optional parameters are self-explanatory.

        If ``test_data`` is provided then the network will be evaluated
        against the test data after each epoch, and partial progress
        printed out. This is useful for tracking progress, but slows things
        down substantially.
        """
        if test_data:
            n_test = len(test_data)
            
        n = len(training_data)

        # For each epoch
        for j in range(epochs):
            time1 = time.time()

            # 1. Shuffle training data randomly
            random.shuffle(training_data)

            # 2. Partition training data into mini-batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # 3. Apply single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

            time2 = time.time()

            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, lr):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``lr``
        is the learning rate.

        Essentially, the gradients for every training example in the
        mini-batch is computed, then used to update the weights and 
        biases.
        """
        # Initializing the gradient vectors of cost function w.r.t
        # biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        X = np.zeros((self.sizes[0], len(mini_batch)))
        y = np.zeros((self.sizes[-1], len(mini_batch)))
        for i in range(len(mini_batch)):
            X[:,i] = mini_batch[i][0].reshape(-1)
            y[:,i] = mini_batch[i][1].reshape(-1)

        delta_nabla_b, delta_nabla_w = self.backprop(X, y)

        nabla_b = [
            nb + dnb.sum(axis=1).reshape(-1, 1) for nb, dnb in zip(nabla_b, delta_nabla_b)
        ]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w-(lr/len(mini_batch))*nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b-(lr/len(mini_batch))*nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, X, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x, given a sample x w.r.t
        the biases and weights.

        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of 
        numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """
        # 0. Initializing the gradient vectors
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 1. Feedforward
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            # It's cool that this works:
            # (30, 784) x (784 x batch-size) + (30, 1) and yet Numpy "knows"
            # to simply add each element in b column-wise
            z = np.dot(w, activation)+b
            # Weighted-input vector per layer
            zs.append(z)

            activation = sigmoid(z)
            # Activations vector per layer
            activations.append(activation)

        # 2. Backward pass
        # Delta at output layer
        # delta : (10 output neurons x mini batch), each column pertains to
        #         a samle in the mini-batch
        delta = self.cost_derivative(
            activations[-1], y
        ) * sigmoid_prime(zs[-1])

        # === === === === === === === === === === === === === === === ===
        # For OUTPUT layer (L)
        # Cost gradient w.r.t bias : (10 output neurons x mini batch), each
        #                            column pertains to a sample in the batch
        nabla_b[-1] = delta

        # Cost gradient w.r.t weight : (# neurons in layer L-1 x # output neurons)
        # -  Interestingsly, this term does not need "special" treatment for
        #    summing over the samples later because of the matrix operation
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # === === === === === === === === === === === === === === === ===

        # Starting from the 2nd LAST layer ...
        for l in range(2, self.num_layers):
            # Get weighted-input at next layer (L-2, L-1, ..., 2)
            z = zs[-l]

            # Multiply weighted-input by the derivative of sigmoid function
            sp = sigmoid_prime(z)

            # Computing the delta at next layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # Appending gradients
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.

        Cost function here is the MSE, C = 0.5*(y - a)^2
        """
        return (output_activations-y)
