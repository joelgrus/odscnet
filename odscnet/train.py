"""
Here's a function to train a neural net
"""

from odscnet.tensor import Tensor
from odscnet.nn import NeuralNet
from odscnet.loss import Loss, MSE
from odscnet.optim import Optimizer, SGD
from odscnet.data import DataIterator, BatchIterator

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predictions = net.forward(batch.inputs)
            epoch_loss += loss.loss(predictions, batch.targets)
            grad = loss.grad(predictions, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
