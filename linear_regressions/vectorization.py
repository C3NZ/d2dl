import math
import time

from mxnet import np

import d2l

total_samples_per_vector = 10000
first_vector = np.ones(total_samples_per_vector)
second_vector = np.ones(total_samples_per_vector)


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def compute_normal_distribution(samples, mean, variance):
    """
    Compute the probability density of a normal distribution.
    https://en.wikipedia.org/wiki/Probability_density_function
    """
    probability = 1 / math.sqrt(2 * math.pi * variance ** 2)
    return probability * np.exp((-0.5 / variance ** 2) * (samples - mean) ** 2)


def plot_normal_distributions():
    """
    Plot normal distributions with different mean (mu) and variance (sigma)
    values to demonstrate what effects different means and variances has on a
    normal distribution.
    """
    # Create an evenly spaced vector from -7 to 7 with 0.01 as the spacing.
    x = np.arange(-7, 7, 0.01)

    # Different parameters to be used for mean and sigma, respectively.
    parameters = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(
        x,
        [
            compute_normal_distribution(x, mean, variance)
            for mean, variance in parameters
        ],
        xlabel="z",
        ylabel="p(z)",
        figsize=(4.5, 2.5),
        legend=[f"mean {mean}, var {variance}" for mean, variance in parameters],
    )
    d2l.plt.savefig("normal_distributions")


def compare_speeds_of_computation():
    """
    Compare the speeds of vector addition through pythonic code and doing the
    addition with np.
    """
    zeros_vector = np.zeros(total_samples_per_vector)
    timer = Timer()

    for i in range(len(zeros_vector)):
        zeros_vector[i] = first_vector[i] + second_vector[i]

    print(f"Pythonic way of doing vector addition: {timer.stop():5f} second(s)")

    timer.start()

    _ = first_vector + second_vector
    print(f"vector addition with np: {timer.stop():5f} second(s)")


def main():
    """
    Execute module code.
    """
    compare_speeds_of_computation()
    plot_normal_distributions()


if __name__ == "__main__":
    main()
