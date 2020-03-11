from mxnet import np, npx
from IPython import display
npx.set_np()

def numerical_func(x):
    return (3 * x ** 2) - (4 * x)


def calculate_numerical_limit(func, x, h):
    return (func(x + h) - func(x)) / h

def find_limit():
    h = 0.1
    for i in range(5):
        numerical_limit = calculate_numerical_limit(numerical_func, 1, h)
        print(f"h = {h:.5f}, numerical limit = {numerical_limit:.5f}")
        h *= .01

if __name__ == "__main__":
    find_limit()
