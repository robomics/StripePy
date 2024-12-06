# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import shutil
import tempfile

import pytest

from stripepy.IO import create_folders_for_plots, remove_and_create_folder


def _directory_is_empty(path) -> bool:
    path = pathlib.Path(path)
    assert path.is_dir()
    return next(path.iterdir(), None) is None  # noqa


def test_folders_for_plots(tmpdir):
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
        test_paths = [
            pathlib.Path(tmpdir),
            pathlib.Path(tmpdir) / "dir2",
        ]

        for test_dir in test_paths:
            if test_dir.exists():
                shutil.rmtree(test_dir)

            assert create_folders_for_plots(test_dir) == [
                test_dir,
                test_dir / "1_preprocessing",
                test_dir / "2_TDA",
                test_dir / "3_shape_analysis",
                test_dir / "4_biological_analysis",
                test_dir / "3_shape_analysis" / "local_pseudodistributions",
            ]


@pytest.mark.unit
class TestRemoveAndCreateFolder:
    # RuntimeError(f"output folder {path} already exists. Pass --force to overwrite it.")
    @staticmethod
    def test_create_new_folder(tmpdir):
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            test_dir = tmpdir / "out"
            remove_and_create_folder(test_dir, force=True)
            assert test_dir.is_dir()
            assert _directory_is_empty(test_dir)

            with pytest.raises(RuntimeError, match="already exists. Pass --force to overwrite it"):
                remove_and_create_folder(test_dir, force=False)

    @staticmethod
    def test_overwrite_existing_folder(tmpdir):
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            test_dir = tmpdir / "out"

            (test_dir / "dir").mkdir(parents=True)
            (test_dir / "file.txt").touch()

            assert not _directory_is_empty(test_dir)
            remove_and_create_folder(test_dir, force=True)
            assert _directory_is_empty(test_dir)


def test_create_folders_for_plots(tmpdir):
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        test_dir = tmpdir / "out"
        result = create_folders_for_plots(test_dir)
        assert isinstance(result, list)
        assert test_dir.is_dir()
