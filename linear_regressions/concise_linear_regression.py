"""
Implementation of a linear regression network using Gluon
"""
from typing import Tuple

from mxnet import autograd, gluon, np, npx

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

    for feature_batch, target_batch in data_iterator:
        print(feature_batch)
        print(target_batch)


if __name__ == "__main__":
    main()
