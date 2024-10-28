import sys

sys.path.insert(0, "./utils")

import pathlib

import pytest
from cli import parse_args as cli_parse_args


class Test_configs_input:

    def test_contact_map(self):
        # Arrange
        argument = "--contact-map"
        variations = ("test\\data\\4DNFI6HDY7WZ.mcool", "test\\data\\4DNFIOTPSS3L.hic")

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", variant, "10000"]
            output.append(cli_parse_args()[0])

        # Assert
        assert output[0] == {
            "contact-map": pathlib.Path("test\\data\\4DNFI6HDY7WZ.mcool"),
            "resolution": 10_000,
            "genomic_belt": 5_000_000,
            "roi": None,
        }

        assert output[1] == {
            "contact-map": pathlib.Path("test\\data\\4DNFIOTPSS3L.hic"),
            "resolution": 10_000,
            "genomic_belt": 5_000_000,
            "roi": None,
        }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        sys.argv = [""]

    def test_resolution(self):
        # Arrange
        argument = "resolution"
        variations = (
            1000,
            2000,
            5000,
            10_000,
            25_000,
            50_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
            2_500_000,
            5_000_000,
            10_000_000,
        )

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", str(variant)]
            output.append(cli_parse_args()[0])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "resolution": variations[testNr],
                "contact-map": pathlib.Path("test\\data\\4DNFI6HDY7WZ.mcool"),
                "genomic_belt": 5_000_000,
                "roi": None,
            }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]

    def test_genomic_belt(self):
        # Arrange
        argument = "--genomic-belt"
        variations = tuple(range(10_000, 1_010_000, 10_000))

        # Act
        output = []
        for variant in variations:
            print("M" + str(variant) + "M")
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, str(variant)]
            output.append(cli_parse_args()[0])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "genomic_belt": variations[testNr],
                "contact-map": pathlib.Path("test\\data\\4DNFI6HDY7WZ.mcool"),
                "resolution": 10_000,
                "roi": None,
            }
        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]

    def test_roi(self):
        # Arrange
        argument = "--roi"
        variations = "middle"

        # Act
        output = []
        sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, variations]
        output.append(cli_parse_args()[0])

        # Assert
        assert output[0] == {
            "roi": "middle",
            "contact-map": pathlib.Path("test\\data\\4DNFI6HDY7WZ.mcool"),
            "resolution": 10_000,
            "genomic_belt": 5_000_000,
        }

        # Cleanup
        argument = None
        variations = None
        output = None
        sys.argv = [""]


class Test_configs_thresholds:

    def test_glob_pers_type(self):
        # Arrange
        argument = "--glob-pers-type"
        variations = ("constant", "adaptive")

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, variant]
            output.append(cli_parse_args()[1])

        # Assert
        assert output[0] == {
            "glob_pers_type": "constant",
            "glob_pers_min": 0.2,
            "constrain_heights": False,
            "loc_pers_min": 0.2,
            "loc_trend_min": 0.1,
            "max_width": 100_000,
        }
        assert output[1] == {
            "glob_pers_type": "adaptive",
            "glob_pers_min": 0.9,
            "constrain_heights": False,
            "loc_pers_min": 0.2,
            "loc_trend_min": 0.1,
            "max_width": 100_000,
        }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        sys.argv = [""]

    def test_glob_pers_min(self):
        # Arrange
        argument = "--glob-pers-min"
        variations = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, str(variant)]
            output.append(cli_parse_args()[1])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "glob_pers_min": variations[testNr],
                "glob_pers_type": "constant",
                "constrain_heights": False,
                "loc_pers_min": 0.2,
                "loc_trend_min": 0.1,
                "max_width": 100_000,
            }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]

    def test_constrain_heights(self):
        # Arrange
        argument = "--constrain-heights"

        # Act
        output = []
        sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000"]
        output.append(cli_parse_args()[1])

        sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument]
        output.append(cli_parse_args()[1])

        # Assert
        assert output[0] == {
            "constrain_heights": False,
            "glob_pers_type": "constant",
            "glob_pers_min": 0.2,
            "loc_pers_min": 0.2,
            "loc_trend_min": 0.1,
            "max_width": 100_000,
        }
        assert output[1] == {
            "constrain_heights": True,
            "glob_pers_type": "constant",
            "glob_pers_min": 0.2,
            "loc_pers_min": 0.2,
            "loc_trend_min": 0.1,
            "max_width": 100_000,
        }

        # Cleanup
        argument = None
        output = None
        sys.argv = [""]

    def test_loc_pers_min(self):
        # Arrange
        argument = "--loc-pers-min"
        variations = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, str(variant)]
            output.append(cli_parse_args()[1])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "loc_pers_min": variations[testNr],
                "glob_pers_type": "constant",
                "glob_pers_min": 0.2,
                "constrain_heights": False,
                "loc_trend_min": 0.1,
                "max_width": 100_000,
            }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]

    def test_loc_trend_min(self):
        # Arrange
        argument = "--loc-trend-min"
        variations = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, str(variant)]
            output.append(cli_parse_args()[1])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "loc_trend_min": variations[testNr],
                "glob_pers_type": "constant",
                "glob_pers_min": 0.2,
                "constrain_heights": False,
                "loc_pers_min": 0.2,
                "max_width": 100_000,
            }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]

    def test_max_width(self):
        # Arrange
        argument = "--max-width"
        variations = tuple(range(10_000, 1_010_000, 10_000))

        # Act
        output = []
        for variant in variations:
            sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, str(variant)]
            output.append(cli_parse_args()[1])

        # Assert
        for testNr in range(len(output)):
            assert output[testNr] == {
                "max_width": variations[testNr],
                "glob_pers_type": "constant",
                "glob_pers_min": 0.2,
                "constrain_heights": False,
                "loc_pers_min": 0.2,
                "loc_trend_min": 0.1,
            }

        # Cleanup
        argument = None
        variations = None
        output = None
        variant = None
        testNr = None
        sys.argv = [""]


class Test_configs_output:
    def test_output_folder(self):
        # Arrange
        argument = "--output-folder"
        variations = "test\\data"

        # Act
        output = []
        sys.argv = ["", "test\\data\\4DNFI6HDY7WZ.mcool", "10000", argument, variations]
        output.append(cli_parse_args()[2])

        # Assert
        assert output[0] == {
            "output_folder": "test\\data",
        }

        # Cleanup
        argument = None
        variations = None
        output = None
        sys.argv = [""]
