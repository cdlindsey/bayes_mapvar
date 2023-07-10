''' Utilities for testing Maximum A Posteriori (MAP) estimation'''

import numpy as np

def reldif(x, y):
    """ Returns max of |x-y|/(|y|+1) and |x-y|/(|x|+1), relative difference

        Parameters:
        -------
        x: 0-2 dimension tensor
        y: 0-2 dimension tensor of same shape as x

        Returns:
        -------
        result: maximum elementwise relative difference between x and y
    """
    # pylint: disable=invalid-name
    if isinstance(x, float) and isinstance(y, float):
        if abs(y) > abs(x):
            return abs(x - y) / (abs(y) + 1)
        else:
            return abs(y - x) / (abs(x) + 1)
    else:
        assert len(x.shape) < 3
        assert len(y.shape) < 3
        assert x.shape == y.shape
        return np.max(
            abs(x - y) / (np.minimum(abs(x), abs(y)) + np.ones(x.shape)))
