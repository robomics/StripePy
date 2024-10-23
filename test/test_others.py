import sys

sys.path.insert(0, "utils/others")

import os
import pathlib
from math import log
from random import randrange
from statistics import median

import hictkpy as htk
import pytest
from others import cmap_loading


class Test_cmap_loading:
    pass


# pathlib.Path("test") / pathlib.Path("data")
paths = (
    pathlib.Path("test") / pathlib.Path("data") / pathlib.Path("4DNFI6HDY7WZ.mcool"),
    pathlib.Path("test") / pathlib.Path("data") / pathlib.Path("4DNFIOTPSS3L.hic"),
)  # TODO: add .cool file

for test_file_paths in paths:
    assert test_file_paths.exists()

resolutions = (
    1000,
    2000,
    5000,
    10000,
    25000,
    50000,
    100000,
    250000,
    500000,
    1000000,
    2500000,
    5000000,
    10000000,
)


def limit_values(iterator):
    iterator = list(iterator)
    limit_values = []
    limit_values.append(iterator.pop(0))
    limit_values.append(iterator.pop(-1))
    limit_values.append(iterator.pop(int(len(iterator) / 2)))

    def bigger_than_zero_smaller_than_max(number, max_value):
        return min(max(number, 0), max_value)

    amountOfValuesToPick = int(log(len(iterator) ** 3 - 8, 2))
    accountForValueLimits = bigger_than_zero_smaller_than_max(amountOfValuesToPick, len(iterator))
    for i in range(accountForValueLimits):
        limit_values.append(iterator.pop(randrange(len(iterator))))
    return set(limit_values)


def arrange():
    pickedResolutions = limit_values(resolutions)
    return pickedResolutions


def XXX():
    results = {}
    for file_types in paths:
        for various_resolutions in resolutions:
            output = cmap_loading(file_types, various_resolutions)
            results[(file_types, various_resolutions)] = output


def test_invalid_path_names():
    # with pytest.raises(Exception, match="Unsupported file format:") as exceptionInfoMcool:
    #    cmap_loading(pathlib.Path(os.listdir()[0]), 5000)
    pytest.raises(Exception, cmap_loading, pathlib.Path(paths[0]), 5000, match="Unsupported file format:")
    pytest.raises(Exception, cmap_loading, pathlib.Path(paths[1]), 5000, match="Unsupported file format:")


## Valid input, erroneous output
def test_invalid_resolutions():
    valid_resolutions = limit_values(resolutions)
    path_mcool = paths[0]
    for test_resolutions in [100, 10**9, 0, -1, -1000]:
        with pytest.raises(Exception, match="Invalid input value:") as exceptionInfo:
            print(path_mcool)
            cmap_loading(path_mcool, test_resolutions)
    assert len(exceptionInfo.vaules.args) == 5


def test_incorrect_implementation():
    pass


def test_correct_output():
    pass


def test_cmap_loading():

    # Assert block with expected chromosomes, chromosome starts, chromosome ends and chromosome sizes
    shouldBeMaxMinAndMedian = {1000, 100_000, 10_000_000}

    for values in shouldBeMaxMinAndMedian:
        assert values in resolutions
    return
