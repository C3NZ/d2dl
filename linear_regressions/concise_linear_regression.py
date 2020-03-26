"""
Implementation of a linear regression network using Gluon
"""
from typing import Tuple

from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import loss, nn

import d2l

npx.set_np()


def load_array(data_arrays: Tuple, batch_size: int, is_training: bool = True):
    """
    Construct a Gluon data loader.
    """
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_training)


def main():
    """
    Main execution of the module.
    """
    # Setup the same initial constraints as our previous linear regression model.
    true_weights = np.array([2, -3.4])
    true_bias = 4.2
    features, targets = d2l.synthetic_data(true_weights, true_bias, 1000)

    batch_size = 10
    data_iterator = load_array((features, targets), batch_size, True)

    # Create a seuqential neural network with one output layer. gluon will infer
    # the input shape the first time data is passed through to it.
    net = nn.Sequential()
    net.add(nn.Dense(1))

    # Initialize the weights with a random sample from a normal distribution
    # with a mean of 0 and a standard deviation of 0.01. bias is initialized as
    # by default. The initialization is deferred until the first attempt to pass
    # data through the network.
    net.initialize(init.Normal(sigma=0.01))

    # The squared loss is also known as the L2 norm loss.
    l2_loss = loss.L2Loss()

    # Setup our SGD optimizer through the trainer class.
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})

    num_epochs = 3

    # Training loop time
    for epoch in range(1, num_epochs + 1):
        for feature_batch, target_batch in data_iterator:
            with autograd.record():
                predicted_targets = net(feature_batch)
                batch_loss = l2_loss(predicted_targets, target_batch)

            batch_loss.backward()
            trainer.step(batch_size)

        epoch_loss = l2_loss(net(features), targets)
        print(f"epoch {epoch}, loss: {epoch_loss.mean().asnumpy()}")


if __name__ == "__main__":
    main()
