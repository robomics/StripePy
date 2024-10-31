import os
import pathlib
import shutil

import pytest


def generate_directory_name():
    directory_name = 0
    while os.path.exists(str(directory_name)):
        directory_name += 1
    directory_name = str(directory_name)
    return directory_name


"""
def _find_directory(dir) -> bool:
    return os.path.exists(dir)
"""


def designated_directory_exists(directory_name) -> bool:
    return os.path.exists(directory_name)
    # return _find_directory(directory_name)


def test_list_folders_for_plots():
    # TODO: Add invalid characters in string cycle.
    from stripepy.IO import list_folders_for_plots

    invalid_types_list = [int, float, bool]
    for types in invalid_types_list:
        with pytest.raises(TypeError) as exceptionFromOne:
            list_folders_for_plots(types(1))
        assert "argument should be a str" in str(exceptionFromOne.value)
        exceptionFromOne = None

        with pytest.raises(TypeError) as exceptionFromZero:
            list_folders_for_plots(types(0))
        assert "argument should be a str" in str(exceptionFromZero.value)
        exceptionFromZero = None
    assert list_folders_for_plots("1"), "Valid data type not accepted in test_list_folders_for_plots"
    assert list_folders_for_plots("0"), "Valid data type not accepted in test_list_folders_for_plots"

    assert list_folders_for_plots("1") == [
        pathlib.Path("1"),
        pathlib.Path("1") / "1_preprocessing",
        pathlib.Path("1") / "2_TDA",
        pathlib.Path("1") / "3_shape_analysis",
        pathlib.Path("1") / "4_biological_analysis",
        pathlib.Path("1") / "3_shape_analysis" / "local_pseudodistributions",
    ]
    assert len(list_folders_for_plots("1")) == 6
    return


def test_remove_and_create_folder():
    from stripepy.IO import remove_and_create_folder

    directory_name = generate_directory_name()

    # Create directory
    remove_and_create_folder(directory_name)
    assert designated_directory_exists(directory_name)

    # Directory already exists
    # TODO: Expand create-delete cycle when test object is given decision input
    remove_and_create_folder(directory_name)
    assert designated_directory_exists(directory_name)

    # Remove directory
    assert designated_directory_exists(directory_name)
    shutil.rmtree(directory_name)
    return


def test_create_folders_for_plots():
    from stripepy.IO import create_folders_for_plots

    directory_name = generate_directory_name()

    assert not designated_directory_exists(directory_name)
    result = create_folders_for_plots(directory_name)
    assert isinstance(result, list)
    assert designated_directory_exists(directory_name)
    shutil.rmtree(directory_name)
    assert not designated_directory_exists(directory_name)
    return


def test_format_ticks():
    pass


def test_HiC():
    pass


def test_pseudodistrib():
    pass


def test_pseudodistrib_and_HIoIs():
    pass


def test_HiC_and_sites():
    pass


def test_HiC_and_HIoIs():
    pass


def test_plot_stripes():
    pass


def test_plot_stripes_and_peaks():
    pass


def test_save_candidates_bedpe():
    pass
