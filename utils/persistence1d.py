import math

import matplotlib.pyplot as plt
import numpy as np
from unionfind import UnionFind

# Implementation adapted from the library by Tino Weinkauf downloadable at:
# https://www.csc.kth.se/~weinkauf/notes/persistence1d.html

# ATT: in its original implementation, the terms minima/maxima/extrema were sometimes used in place of
# minimum/maximum/extremum points. This notation is here fixed.
# For more details, consult: https://en.wikipedia.org/wiki/Maximum_and_minimum


def RunPersistence(InputData, levelsets="lower"):
    """
    This function share its name to Weinkauf's implementation, but can here work on upper level sets.

    Finds extrema and their persistence in one-dimensional data w.r.t. lower (default, levelsets="lower")
    or upper (levelsets="upper") level sets.

    Local minima and local maxima are extracted, paired, and returned together with their persistence.
    For levelsets="lower", the global minimum is extracted as well.
    For levelsets="upper", the global maximum is extracted as well.

    We assume a connected one-dimensional domain.

    Short explanation for the case of levelsets=="lower" (the case of levelsets="upper") is analogous.
    Think of "data on a line", or a function f(x) over some domain xmin <= x <= xmax. We are only concerned with the
    data values f(x) and do not care to know the x positions of these values, since this would not change which
    point is a minimum or maximum.

    This function returns a list of extrema together with their persistence. The list is NOT sorted, but the paired
    extrema can be identified, i.e., which minimum and maximum were removed together at a particular persistence
    level. As follows:
    (*)  The odd entries are minima, the even entries are maxima.
    (*)  The minimum at 2*i is paired with the maximum at 2*i+1.
    (*)  The last entry of the list is the global minimum (resp. maximum) when levelsets="lower" (resp.
        levelsets="upper"). It is not paired with a maximum (resp. minimum).
    Hence, the list has an odd number of entries.

    Authors: Tino Weinkauf (original implementation) and Andrea Raffo (modified implementation)
    """

    # ~ How many items do we have?
    NumElements = len(InputData)

    # ~ Sort data in a stable manner to break ties (leftmost index comes first)
    if levelsets == "lower":
        SortedIdx = np.argsort(InputData, kind="stable")
    elif levelsets == "upper":
        SortedIdx = np.argsort(InputData, kind="stable")[::-1]

    # ~ Get a union find data structure
    UF = UnionFind(NumElements)

    # ~ Paired extrema
    ExtremumPointsAndPersistence = []

    # ~ Watershed
    for idx in SortedIdx:

        # ~ Get neighborhood indices
        LeftIdx = max(idx - 1, 0)
        RightIdx = min(idx + 1, NumElements - 1)

        # ~ Count number of components in neighborhhood
        NeighborComponents = []
        LeftNeighborComponent = UF.Find(LeftIdx)
        RightNeighborComponent = UF.Find(RightIdx)
        if LeftNeighborComponent != UnionFind.NOSET:
            NeighborComponents.append(LeftNeighborComponent)
        if RightNeighborComponent != UnionFind.NOSET:
            NeighborComponents.append(RightNeighborComponent)

        # ~ Left and Right cannot be the same set in a 1D domain
        NumNeighborComponents = len(NeighborComponents)

        if NumNeighborComponents == 0:
            # ~ Create a new component
            UF.MakeSet(idx)
        elif NumNeighborComponents == 1:
            # ~ Extend the one and only component in the neighborhood ~ Note that NeighborComponents[0] holds the
            # root of a component, since we called Find() earlier to retrieve it
            UF.ExtendSetByID(NeighborComponents[0], idx)
        else:
            if levelsets == "lower":

                # ~ Merge the two components on either side of the current point
                idxLowestNeighborComp = np.argmin(InputData[NeighborComponents])
                idxLowestMinimum = NeighborComponents[idxLowestNeighborComp]
                idxHighestMinimum = NeighborComponents[(idxLowestNeighborComp + 1) % 2]
                UF.ExtendSetByID(idxLowestMinimum, idx)
                UF.Union(idxHighestMinimum, idxLowestMinimum)

                # ~ Record the two paired extrema: index of minimum, index of maximum, persistence value
                Persistence = InputData[idx] - InputData[idxHighestMinimum]
                ExtremumPointsAndPersistence.append((idxHighestMinimum, Persistence))
                ExtremumPointsAndPersistence.append((idx, Persistence))

            elif levelsets == "upper":

                # ~ Merge the two components on either side of the current point
                idxHighestNeighborComp = np.argmax(InputData[NeighborComponents])
                idxHighestMaximum = NeighborComponents[idxHighestNeighborComp]
                idxLowestMaximum = NeighborComponents[(idxHighestNeighborComp + 1) % 2]
                UF.ExtendSetByID(idxHighestMaximum, idx)
                UF.Union(idxLowestMaximum, idxHighestMaximum)

                # ~ Record the two paired extrema: index of minimum, index of maximum, persistence value
                Persistence = InputData[idxLowestMaximum] - InputData[idx]
                ExtremumPointsAndPersistence.append((idx, Persistence))
                ExtremumPointsAndPersistence.append((idxLowestMaximum, Persistence))

    # ~ Global minimum (or maximum)
    if levelsets == "lower":
        idxGlobalMinimum = UF.Find(0)
        ExtremumPointsAndPersistence.append((idxGlobalMinimum, np.inf))
    elif levelsets == "upper":
        idxGlobalMaximum = UF.Find(0)
        ExtremumPointsAndPersistence.append((idxGlobalMaximum, np.inf))

    return ExtremumPointsAndPersistence


def DiversifyExtremumPointsAndPersistence(ExtremumPointsAndPersistence, level_set):
    MinimumPointsAndPersistence = [t for t in ExtremumPointsAndPersistence[::2]]
    MaximumPointsAndPersistence = [t for t in ExtremumPointsAndPersistence[1::2]]

    if level_set == "upper":
        MaximumPointsAndPersistence = MaximumPointsAndPersistence + [MinimumPointsAndPersistence[-1]]
        MinimumPointsAndPersistence = MinimumPointsAndPersistence[:-1]

    return MinimumPointsAndPersistence, MaximumPointsAndPersistence


def FilterExtremumPointsByPersistence(ExtremumPointsAndPersistence, Threshold):

    FilteredExtremumPointsAndPersistence = [t for t in ExtremumPointsAndPersistence if t[1] > Threshold]
    return FilteredExtremumPointsAndPersistence


def plot_persistence(
    birth_levels,
    death_levels,
    thresh_birth_levels,
    thresh_death_levels,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    """
    Plot persistence pairs, i.e., (birth_level, death_level) pair of each maximum.
    :param birth_levels:
    :param death_levels:
    :param thresh_birth_levels:
    :param thresh_death_levels:
    :param output_folder:
    :param file_name:
    :param title:
    :param display:
    :return: -
    """

    # Setup figure
    fig, ax = plt.subplots(1, 1)

    # Plot the persistence
    plt.scatter(birth_levels, death_levels, marker=".", linewidths=1, color="red", label="discarded")
    plt.scatter(thresh_birth_levels, thresh_death_levels, marker=".", linewidths=1, color="blue", label="selected")

    X = np.c_[birth_levels, death_levels]
    ax.plot([0, 1], [0, 1], "-", c="grey")
    ax.set_xlabel("Birth level")
    ax.set_ylabel("Death level")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.axis("scaled")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if output_folder is not None and file_name is not None:
        plt.savefig(output_folder + "/" + file_name)

    if display is True:
        plt.show()
    else:
        plt.close()
