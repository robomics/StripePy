import numpy as np
from persistence1d import (
    DiversifyExtremumPointsAndPersistence,
    FilterExtremumPointsByPersistence,
    RunPersistence,
    plot_persistence,
)


def TDA(marginal_pd, min_persistence=None):
    # Compute the extremum points (i.e., minimum and maximum points) of the marginal pseudo-distribution AND their
    # persistence:
    ExtremumPointsAndPersistence = RunPersistence(marginal_pd, levelsets="upper")

    # Keep only those extremum points with persistence above a given value:
    ThreshExtremumPointsAndPersistence = FilterExtremumPointsByPersistence(
        ExtremumPointsAndPersistence, Threshold=min_persistence
    )

    # Split extremum points into minimum and maximum points:
    (MinimumPointsAndPersistence, MaximumPointsAndPersistence) = DiversifyExtremumPointsAndPersistence(
        ExtremumPointsAndPersistence, level_set="upper"
    )
    (ThreshMinimumPointsAndPersistence, ThreshMaximumPointsAndPersistence) = DiversifyExtremumPointsAndPersistence(
        ThreshExtremumPointsAndPersistence, level_set="upper"
    )

    # Sorting maximum points (and, as a consequence, the corresponding minimum points) w.r.t. persistence:
    argsorting = np.argsort(list(zip(*MaximumPointsAndPersistence))[1]).tolist()
    argsorting_thresh = np.argsort(list(zip(*ThreshMaximumPointsAndPersistence))[1]).tolist()

    if len(ThreshMinimumPointsAndPersistence) == 0:
        ThreshMinimumPoints = []
        pers_of_ThreshMinimumPoints = []
    else:
        ThreshMinimumPoints = np.array(list(zip(*ThreshMinimumPointsAndPersistence))[0])[
            argsorting_thresh[:-1]
        ].tolist()
        pers_of_ThreshMinimumPoints = np.array(list(zip(*ThreshMinimumPointsAndPersistence))[1])[
            argsorting_thresh[:-1]
        ].tolist()

    # Indices of maximum points and their persistence:
    ThreshMaximumPoints = np.array(list(zip(*ThreshMaximumPointsAndPersistence))[0])[argsorting_thresh].tolist()
    pers_of_ThreshMaximumPoints = np.array(list(zip(*ThreshMaximumPointsAndPersistence))[1])[argsorting_thresh].tolist()

    if len(MinimumPointsAndPersistence) > 0:
        # Indices of minimum points and their persistence:
        MinimumPoints = np.array(list(zip(*MinimumPointsAndPersistence))[0])[argsorting[:-1]].tolist()
        # pers_of_MinimumPoints = np.array(list(zip(*MinimumPointsAndPersistence))[1])[argsorting[:-1]].tolist()
        MaximumPoints = np.array(list(zip(*MaximumPointsAndPersistence))[0])[argsorting].tolist()
        # pers_of_MaximumPoints = np.array(list(zip(*MaximumPointsAndPersistence))[1])[argsorting].tolist()

        # # Plot persistence diagram (NB: global maximum is NOT included)
        # if output_folder is not None:
        #     plot_persistence(
        #         marginal_pd[MaximumPoints[:-1]], marginal_pd[MinimumPoints],
        #         marginal_pd[ThreshMaximumPoints[:-1]], marginal_pd[ThreshMinimumPoints],
        #         output_folder=output_folder, file_name=file_name, title=title, display=display)

    return ThreshMinimumPoints, pers_of_ThreshMinimumPoints, ThreshMaximumPoints, pers_of_ThreshMaximumPoints
