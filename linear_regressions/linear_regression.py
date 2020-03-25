import random

from mxnet import autograd, np, npx

import d2l

npx.set_np()


def generate_synthetic_data(weights, bias, num_examples):
    """
    Generate synthetic data that represents:
    y = X*w + b + noise
    """
    # Create a random noraml distribution between 0 and 1 that have a shape of
    # num_examples multiplied by the number of weights.j
    features = np.random.normal(0, 1, (num_examples, len(weights)))

    # Compute the real linear regression.
    targets = np.dot(features, weights) + bias

    # Add noise to all of our targets with a standard deviation of at most
    # 0.01, making our problem relatively easy.
    targets += np.random.normal(0, 0.01, targets.shape)

    return features, targets


def main():
    """
    Execute main functions of this module.
    """
    true_weights = np.array([2, -3.4])
    true_bias = 4.2

    features, labels = generate_synthetic_data(true_weights, true_bias, 1000)

    print(features)
    print(labels)


if __name__ == "__main__":
    main()
