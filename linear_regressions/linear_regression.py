import random
from typing import List, Tuple

from mxnet import autograd, np, npx

import d2l

npx.set_np()


def generate_synthetic_data(
    weights: np.array, bias: float, num_examples: int
) -> Tuple[np.array, np.array]:
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


def plot_generated_data_points(features: np.array, targets: np.array) -> None:
    """
    Plot our features and targets in a scatter plot.
    """
    d2l.set_figsize((3.5, 2.5))
    d2l.plt.scatter(features[:, 1].asnumpy(), targets.asnumpy(), 1)
    d2l.plt.savefig("generated_data")


def read_data_in_batches(
    batch_size: int, features: np.array, targets: np.array
) -> Tuple[np.array, np.array]:
    """
    Read data previously generated in minibatches for training.
    """
    total_num_samples = len(features)
    indices = list(range(total_num_samples))
    random.shuffle(indices)

    for i in range(0, total_num_samples, batch_size):
        batch_indices = np.array(indices[i : min(i + batch_size, total_num_samples)])
        yield features[batch_indices], targets[batch_indices]


def compute_linear_regression(
    features: np.array, weights: np.array, bias: int
) -> np.array:
    """
    Linear regression implementation.
    Equal to: X*w + b
    """
    return np.dot(features, weights) + bias


def compute_squared_loss(
    predicted_targets: np.array, measured_targets: np.array
) -> np.array:
    """
    The loss function.
    Equal to 1/2 * (y_pred - y_true) ** 2
    """
    # Reshape the measured results to be the same dimension as the predicted
    # results.
    reshaped_measured_targets = measured_targets.reshape(predicted_targets.shape)
    return (predicted_targets - reshaped_measured_targets) ** 2 / 2


def stochastic_gradient_descent(
    trainable_parameters: List[np.array], learning_rate: float, batch_size: int
) -> None:
    """
    Estimate the gradient of the loss with respect to our parameters. We apply
    a SGD update given the trainable parameters, learning rate, and batch size.
    The size of the update is determined by the learning rate and normalized by
    the batch size, so that the magnitude of a typical size step doesn't depend
    on the batch size we choose.

    Will update the parameters in place.
    """
    for parameter in trainable_parameters:
        parameter[:] = parameter - (learning_rate * parameter.grad / batch_size)


def execute_model() -> None:
    """
    Create our linear regression model.
    """
    # Our initial random weights and bias
    trainable_weights = np.random.normal(0, 0.01, (2, 1))
    trainable_bias = np.zeros(1)

    # Attach gradients to each for taking the partial derivatives of our our loss
    # function with respect to these variables. (autograd will take care of the
    # actual computation).
    trainable_weights.attach_grad()
    trainable_bias.attach_grad()

    # Setup synthetically generated data.
    true_weights = np.array([2, -3.4])
    true_bias = 4.2
    features, targets = generate_synthetic_data(true_weights, true_bias, 1000)

    # setup model hyperparameters
    learning_rate = 0.03
    num_epochs = 3
    batch_size = 10

    for epoch in range(num_epochs):
        for feature_batch, target_batch in read_data_in_batches(
            batch_size, features, targets
        ):
            # Record the gradients of the current batch loss.
            with autograd.record():
                # Compute targets given the current features, weights, and bias.
                predicted_targets = compute_linear_regression(
                    feature_batch, trainable_weights, trainable_bias
                )
                # Measure the loss of the predictions against the real values.
                loss = compute_squared_loss(predicted_targets, target_batch)

            # Compute the gradients of our loss with respect to the trainable
            # weights and bias
            loss.backward()

            # Update model weights and bias now since the gradients have been
            # recorded.
            stochastic_gradient_descent(
                [trainable_weights, trainable_bias], learning_rate, batch_size
            )

        # Compute the overall results for the current epoch.
        all_predicted_targets = compute_linear_regression(
            features, trainable_weights, trainable_bias
        )
        total_loss = compute_squared_loss(all_predicted_targets, targets)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss {total_loss.mean().asnumpy()}")

    # Compute the error between our obtained weights/bias against the true
    # weights/bias.
    measured_error_for_weights = true_weights - trainable_weights.reshape(
        true_weights.shape
    )
    measured_error_for_bias = true_bias - trainable_bias
    print(f"Error in estimating weights: {measured_error_for_weights}")
    print(f"Error in estimating bias: {measured_error_for_bias}")


def main():
    """
    Execute main functions of this module.
    """
    execute_model()


if __name__ == "__main__":
    main()
