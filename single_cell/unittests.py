# Kai Sandbrink
# 2022-01-29
# Some unittests for simple functions

import unittest

from rowwise_neuron_curves import compute_dist_metric
import numpy as np

class TestRowwiseNeuronCurves(unittest.TestCase):

    def test_distance_metric(self):
        A = np.array([[1, 1, 1],
            [2, 2, 2]])

        B = np.array([[1, 2, 1],
            [2, 2, 1]])

        self.assertEqual(compute_dist_metric(A, B), 1)

if __name__ == '__main__':
    unittest.main()
