import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(1)
np.random.seed(1)

def mse(pred, target):
    return ((pred - target) ** 2).mean()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Layer(object): pass

class Linear(Layer):
    def __init__(self, in_dim, out_dim, activation, derivative):
        self.weights    = np.random.random((in_dim, out_dim))
        self.activation = activation
        self.derivative = derivative
    def deriv(self):
        return self.derivative(self.last)
    def __call__(self, x):
        self.last = self.activation(x @ self.weights)
        return self.last
    def backward(self, loss):
        pass

class NN(object):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(
            in_dim,
            hidden_dim,
            activation=sigmoid,
            derivative=sigmoid_derivative)
        self.fc2 = Linear(
            hidden_dim,
            out_dim,
            activation=sigmoid,
            derivative=sigmoid_derivative)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    def backward(self, target, pred, loss_fn, lr):
        # Calculate loss
        loss = loss_fn(target, pred)

        # Get layers
        layers = [self.__dict__[attr] for attr in self.__dict__
                  if issubclass(type(self.__dict__[attr]), Layer)]

        # Calculate deltas
        deltas = [0] * len(layers) # last to first
        print("layers:", layers)
        for i in reversed(range(len(layers))):
            cur_layer = layers[i]
            print("backprop:", i)
            
            if i == len(layers) - 1:
                last_error = loss
                last_delta = last_error * cur_layer.deriv()
                deltas[i] = last_delta
            else:
                prev_layer  = layers[i+1]
                prev_delta  = deltas[i+1]
                other_error = prev_delta.dot(prev_layer.weights.T)
                other_delta = other_error * cur_layer.deriv()
                deltas[i]   = other_delta
        
        # Update weights
        for i in range(len(layers)):
            if i == 0:
                layers[i].weights += target.T.dot(deltas[i]) * lr
            else:
                layers[i].weights += layers[i-1].last.T.dot(deltas[i]) * lr

    def backward_old(self, target, pred, loss_fn, lr):
        loss = loss_fn(target, pred)

        fc2_delta = loss * sigmoid_derivative(sigmoid(self.fc2.last))
        fc1_error = fc2_delta.dot(self.fc2.weights.T)
        fc1_delta = fc1_error * sigmoid_derivative(sigmoid(self.fc1.last))

        self.fc1.weights += target.T.dot(fc1_delta) * lr
        self.fc2.weights += sigmoid(self.fc1.last).T.dot(fc2_delta) * lr

    def __call__(self, x):
        return self.forward(x)

if __name__ == "__main__":
    X_s = np.array([[1],
                  [1]], dtype=np.float32)
    
    y_s = np.array([[1],
                  [1]], dtype=np.float32)

    lr = 1e-1
    model = NN(in_dim=1, hidden_dim=1, out_dim=1)

    losses = []
    epochs = 1000
    for epoch_idx in range(epochs):
        batch_idx = 0
        x = X_s
        y = y_s
        pred = model(x)
        loss = mse(pred, y)
        print("Epoch,Batch:", epoch_idx+1, batch_idx+1, x, pred, y, loss)
        losses.append(loss)
        model.backward(target=x, pred=pred, loss_fn=mse, lr=lr)
        batch_idx += 1

    plt.plot(losses)
    plt.show()