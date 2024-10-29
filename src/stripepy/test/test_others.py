import sys

sys.path.insert(0, "utils/others")

import os
import pathlib
from math import log
from random import randrange
from statistics import median

import hictkpy as htk
import pytest

from stripepy.others import cmap_loading


class Test_cmap_loading:
    # pathlib.Path("test") / pathlib.Path("data")
    def generate_paths_to_test_files(self):
        paths = (
            pathlib.Path("test") / pathlib.Path("data") / pathlib.Path("4DNFI6HDY7WZ.mcool"),
            pathlib.Path("test") / pathlib.Path("data") / pathlib.Path("4DNFIOTPSS3L.hic"),
        )  # TODO: add .cool file

        for testFilePaths in paths:
            assert os.path.exists(testFilePaths)

        return paths

    def shared_resolutions():
        sharedResolutions = [
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
        ]
        return sharedResolutions

    def limit_values(self, iterator):
        iterator = list(iterator)
        limitValues = []
        limitValues.append(iterator.pop(0))
        limitValues.append(iterator.pop(-1))
        limitValues.append(iterator.pop(int(len(iterator) / 2)))

        def bigger_than_zero_smaller_than_max(number, max_value):
            return min(max(number, 0), max_value)

        amountOfValuesToPick = int(log(len(iterator) ** 3 - 8, 2))
        accountForValueLimits = bigger_than_zero_smaller_than_max(amountOfValuesToPick, len(iterator))
        for i in range(accountForValueLimits):
            limitValues.append(iterator.pop(randrange(len(iterator))))
        return set(limitValues)

    """
    def pickResolutions(self):
        pickedResolutions = self.limit_values(resolutions)
        return pickedResolutions
    """

    def test_invalid_path_names(self):
        # Arrange
        # paths = self.generate_paths_to_test_files()

        # Act
        with pytest.raises(Exception, match="Unsupported file format:") as exceInfoThisFile:
            cmap_loading("test_others.py", 5000)

        with pytest.raises(Exception, match="Unsupported file format:") as exceInfoDirectory:
            cmap_loading("./data", 5000)

        with pytest.raises(Exception, match="Unsupported file format:") as exceInfoNonexistent:
            cmap_loading("nonexistentFile.txt", 5000)

        # Assert
        assert exceInfoThisFile.value
        assert exceInfoDirectory.value
        assert exceInfoNonexistent.value

        # Cleanup
        exceInfoThisFile = None
        exceInfoDirectory = None
        exceInfoNonexistent = None

    def test_valid_path_names(self):
        # Arrange
        paths = self.generate_paths_to_test_files()

        # Act
        validPathMcool = cmap_loading(str(paths[0]), 1000)
        validPathHic = cmap_loading(str(paths[1]), 1000)

        # Assert
        assert validPathMcool
        assert validPathHic

        # Cleanup
        validPathMcool = None
        validPathHic = None

    ## Valid input, erroneous output
    def test_invalid_resolutions(self):
        # Arrange
        paths = self.generate_paths_to_test_files()

        # Act
        for testResolutionsMcool in [100, 10**9, 0, -1, -1000]:
            with pytest.raises(Exception, match="Invalid input value:") as exceptionInfoMcool:
                cmap_loading(pathMcool := str(paths[0]), testResolutionsMcool)

        for testResolutionsHic in [100, 10**9, 0, -1, -1000]:
            with pytest.raises(Exception, match="Invalid input value:") as exceptionInfoHic:
                cmap_loading(pathHic := str(paths[1]), testResolutionsHic)
        # Assert
        assert len(exceptionInfoMcool.vaules.args) == 5
        assert len(exceptionInfoHic.values.args) == 5

        # Cleanup
        testResolutionsMcool = None
        testResolutionsHic = None
        pathMcool = None
        pathHic = None
        exceptionInfoMcool = None
        exceptionInfoHic = None

    def test_incorrect_implementation(self):
        # Arrange
        # Act
        # Assert
        # Cleanup
        pass

    def test_correct_output(self):
        # Arrange
        paths = self.generate_paths_to_test_files()
        # resolutions = self.limit_values(self.shared_resolutions())
        resolutions = (1000, 25000, 100_000, 500_000, 10_000_000)

        def _expectOutput(chromosomeLengths):
            startList = [0]
            for startCount in chromosomeLengths:
                startList.append((startList[-1] + startCount))
            startList.pop(-1)

            endList = [0]
            for endCount in chromosomeLengths:
                endList.append(endList[-1] + ceil(endCount / resolution))
            endList.pop(0)

            chr_sizes = chromosomeLengths

            return False, startList, endList, chr_sizes

        def _compare(iteratorA, iteratorB):
            # assert iteratorA[0] == iteratorB[0]
            assert iteratorA[1] == iteratorB[1]
            assert iteratorA[2] == iteratorB[2]
            assert iteratorA[3] == iteratorB[3]

        # Act
        results = {}

        for path in paths:
            for resolution in resolutions:
                results[(str(path), resolution)] = cmap_loading(str(path), resolution)
        for item in results.items():
            print(item)
            print("\n\n")
        # Assert
        ## mcool
        ### res 1000
        resultsMcool1000 = results[("test\\data\\4DNFI6HDY7WZ.mcool", 1000)]
        expectedOutputMcool1000 = _expectOutput(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )

        # assert resultsMcool1000[0] == expectedOutputMcool1000[0]

        assert resultsMcool1000[1] == expectedOutputMcool1000[1]

        assert resultsMcool1000[2] == expectedOutputMcool1000[2]

        assert resultsMcool1000[3] == expectedOutputMcool1000[3]

        ### res 25000
        resultsMcool25000 = results["test\\data\\4DNFI6HDY7WZ.mcool", 25000]
        expectedOutputMcool25000 = _expectOutput(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )

        # assert resultsMcool25000[0] == expectedOutputMcool25000[0]
        assert resultsMcool25000[1] == expectedOutputMcool25000[1]
        assert resultsMcool25000[2] == expectedOutputMcool25000[2]
        assert resultsMcool25000[3] == expectedOutputMcool25000[3]

        # res 100_000
        resultsMcool100_000 = results["test\\data\\4DNFI6HDY7WZ.mcool", 100_000]
        expectedOutputMcool100_000 = _expectOutput(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )

        # assert resultsMcool100_000[0] == expectedOutputMcool100_000[0]
        assert resultsMcool100_000[1] == expectedOutputMcool100_000[1]
        assert resultsMcool100_000[2] == expectedOutputMcool100_000[2]
        assert resultsMcool100_000[3] == expectedOutputMcool100_000[3]

        _compare(resultsMcool100_000, expectedOutputMcool100_000)

        ### res 500_000
        resultsMcool500_000 = results["test\\data\\4DNFI6HDY7WZ.mcool", 500_000]
        expectedOutputMcool500_000 = _expectOutput(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )
        _compare(resultsMcool500_000, expectedOutputMcool500_000)

        ### res 10_000_000
        resultsMcool10_000_000 = results["test\\data\\4DNFI6HDY7WZ.mcool", 10_000_000]
        expectedOutputMcool10_000_000 = _expectOutput(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )
        _compare(resultsMcool10_000_000, expectedOutputMcool10_000_000)

        ## hic
        ### res 1000
        resultsHic1000 = results["test\\data\\4DNFIOTPSS3L.hic", 1000]
        expectedOutputHic1000 = _expectOutput(
            [23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352], 1000
        )
        _compare(resultsHic1000, expectedOutputHic1000)

        ### res 25000
        resultsHic25000 = results["test\\data\\4DNFIOTPSS3L.hic", 25000]
        expectedOutputHic25000 = _expectOutput(
            [23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352], 25000
        )
        _compare(resultsHic25000, expectedOutputHic25000)

        ### res 100_000
        resultsHic100_000 = results["test\\data\\4DNFIOTPSS3L.hic", 100_000]
        expectedOutputHic100_000 = _expectOutput(
            [23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352], 100_000
        )
        _compare(resultsHic100_000, expectedOutputHic100_000)

        ### res 500_000
        resultsHic500_000 = results["test\\data\\4DNFIOTPSS3L.hic", 500_000]
        expectedOutputHic500_000 = _expectOutput(
            [23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352], 500_000
        )
        _compare(resultsHic500_000, expectedOutputHic500_000)

        ### res 10_000_000
        resultshic10_000_000 = results["test\\data\\4DNFIOTPSS3L.hic", 10_000_000]
        expectedOutputHic10_000_000 = _expectOutput(
            [23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352], 10_000_000
        )
        _compare(resultshic10_000_000, expectedOutputHic10_000_000)

        # Cleanup

        pass

    def test_input_corrupted_file(self):
        # Arrange
        # Act
        # Assert
        # Cleanup
        pass
