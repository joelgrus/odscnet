"""
fizz buzz is the following problem

given an input x,
if x % 3 == 0, print "fizz"
if x % 5 == 0, print "buzz"
if x % 15 == 0, print "fizzbuzz"
otherwise just print x
"""
from typing import List

import numpy as np

from odscnet.train import train
from odscnet.nn import NeuralNet
from odscnet.layers import Linear, Tanh
from odscnet.optim import SGD

def binary_encode(x: int) -> List[int]:
    """
    return x as a 10-digit binary number
    """
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

inputs = np.array([
    binary_encode(x) for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x) for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net, inputs, targets, num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    inputs = binary_encode(x)
    prediction = net.forward(inputs)
    actual = fizz_buzz_encode(x)
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    prediction_idx = np.argmax(prediction)
    actual_idx = np.argmax(actual)

    print(x, labels[prediction_idx], labels[actual_idx])
