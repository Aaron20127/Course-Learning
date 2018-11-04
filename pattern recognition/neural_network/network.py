# %load network.py

"""
最基本的神经网络结构，《深度学习与神经网络》的源码：
python2.7: https://github.com/mnielsen/neural-networks-and-deep-learning
python3.5: https://github.com/MichalDanielDobrzanski/DeepLearningPython35
"""

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, stop=(0,0)):

        stop_count = 0
        last_accuracy = 0
        max_accuracy = 0
        accuracy = 0
        training_accuracy = 0
        last_epoch = 0


        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                training_accuracy = self.evaluate_training(training_data) * 100.0 / n
                correct = self.evaluate(test_data)
                accuracy = correct * 100.0 / n_test 
                if accuracy > max_accuracy:
                    max_accuracy = accuracy

                print("\nEpoch {} ".format(j));
                print("training accuracy: {:.2f}%".format(training_accuracy));
                print("test correct : {} / {}".format(correct, n_test));
                print("test accuracy: {:.2f}%".format(accuracy));

                ## 测试稳定stop次后停止
                if stop[0] > 0:
                    if (abs(last_accuracy - accuracy) < stop[1]):
                        stop_count += 1
                        if stop_count >= stop[0]:
                            break;
                    else:
                        stop_count = 0
                last_accuracy = accuracy
                print("stop_count: {} ".format(stop_count));

            else:
                print("Epoch {} complete".format(j))

        print("\nTraining complete")
        print("max accuracy: {:.2f}%".format(max_accuracy));
        return accuracy, training_accuracy, last_epoch

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate_training(self, training_data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in training_data]
        return sum(int(x == y) for (x, y) in results)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
