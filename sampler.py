import numpy as np
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt


def sampler(number_of_snapshots, iNumDimensions, use_LHS, do_plot=False):
    if not use_LHS:
        xy_min = -np.ones(iNumDimensions)
        xy_max = np.ones(iNumDimensions)
        SampledPoints = np.random.uniform(
            low=xy_min, high=xy_max, size=(number_of_snapshots, len(xy_min))
        )
    else:
        # using LHS from smt
        limits_ones = np.ones((iNumDimensions, 2))
        limits_ones[:, 0] = -1
        # Create a sampling method
        sampling_method_all = LHS(xlimits=limits_ones)
        # Generate the sampling points
        SampledPoints = sampling_method_all(number_of_snapshots)
    # plot values
    if do_plot:
        plt.plot(SampledPoints[:, 0], SampledPoints[:, 1], "+")
        plt.show()
    return SampledPoints
