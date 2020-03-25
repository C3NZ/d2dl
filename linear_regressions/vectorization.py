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


def main():
    zeros_vector = np.zeros(total_samples_per_vector)
    timer = Timer()

    for i in range(len(zeros_vector)):
        zeros_vector[i] = first_vector[i] + second_vector[i]

    print(f"Pythonic way of doing vector addition: {timer.stop():5f} second(s)")

    timer.start()

    new_vector = first_vector + second_vector
    print(f"vector addition with np: {timer.stop():5f} second(s)")


if __name__ == "__main__":
    main()
