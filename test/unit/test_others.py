# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import tempfile
from typing import List

import pytest

from stripepy.others import open_matrix_file_checked

from .common.cool import generate_singleres_test_file


def _discretize(v: List[int], factor: int) -> List[int]:
    assert factor > 0
    return [(x + factor - 1) // factor for x in v]


@pytest.mark.unit
class TestOpenMatrixFileChecked:
    def test_invalid_paths(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        with pytest.raises(RuntimeError):
            open_matrix_file_checked(tmpdir / "foo", 5000)

    def test_invalid_formats(self, tmpdir):
        invalid_file = pathlib.Path(__file__).resolve()
        folder = pathlib.Path(tmpdir)
        with pytest.raises(RuntimeError):
            open_matrix_file_checked(invalid_file, 5000)

        with pytest.raises(RuntimeError):
            open_matrix_file_checked(folder, 5000)

    def test_invalid_resolutions(self, tmpdir):
        with tempfile.NamedTemporaryFile(dir=tmpdir) as clr:
            clr.close()
            path_to_clr = generate_singleres_test_file(pathlib.Path(clr.name), 1000)

            with pytest.raises(RuntimeError):
                open_matrix_file_checked(path_to_clr, 5000)
            with pytest.raises(TypeError, match="must be an integer"):
                open_matrix_file_checked(path_to_clr, 1000.0)
            with pytest.raises(ValueError, match="must be greater than zero"):
                open_matrix_file_checked(path_to_clr, -1)
            with pytest.raises(ValueError, match="must be greater than zero"):
                open_matrix_file_checked(path_to_clr, 0)

    def test_valid_files(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        chromosomes = {"A": 100_000, "B": 50_000, "C": 10_000}
        resolution = 1000
        path_to_clr = generate_singleres_test_file(tmpdir / "test.cool", resolution, chromosomes)

        f = open_matrix_file_checked(path_to_clr, resolution)
        assert f.resolution() == 1000

        with pytest.raises(Exception):
            open_matrix_file_checked(path_to_clr, resolution + 1)
