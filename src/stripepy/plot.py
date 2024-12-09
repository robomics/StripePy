# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import warnings
from typing import Dict, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import EngFormatter, ScalarFormatter


@functools.cache
def _get_custom_palettes() -> Dict[str, npt.NDArray]:
    # Source: https://github.com/open2c/cooltools/blob/master/cooltools/lib/plotting.py
    return {
        "fall": np.array(
            (
                (255, 255, 255),
                (255, 255, 204),
                (255, 237, 160),
                (254, 217, 118),
                (254, 178, 76),
                (253, 141, 60),
                (252, 78, 42),
                (227, 26, 28),
                (189, 0, 38),
                (128, 0, 38),
                (0, 0, 0),
            )
        )
        / 255,
        "fruit_punch": np.array(
            (
                (255, 255, 255),
                (255, 204, 204),
                (255, 153, 153),
                (255, 102, 102),
                (255, 50, 50),
                (255, 0, 0),
            )
        )
        / 255,
        "blues": np.array(
            (
                (255, 255, 255),
                (180, 204, 225),
                (116, 169, 207),
                (54, 144, 192),
                (5, 112, 176),
                (4, 87, 135),
                (3, 65, 100),
                (2, 40, 66),
                (1, 20, 30),
                (0, 0, 0),
            )
        )
        / 255,
        "acidblues": np.array(
            (
                (255, 255, 255),
                (162, 192, 222),
                (140, 137, 187),
                (140, 87, 167),
                (140, 45, 143),
                (120, 20, 120),
                (90, 15, 90),
                (60, 10, 60),
                (30, 5, 30),
                (0, 0, 0),
            )
        )
        / 255,
        "nmeth": np.array(
            (
                (236, 250, 255),
                (148, 189, 217),
                (118, 169, 68),
                (131, 111, 43),
                (122, 47, 25),
                (41, 0, 20),
            )
        )
        / 255,
    }


def _list_to_colormap(color_list, name=None) -> mpl.colors.LinearSegmentedColormap:
    color_list = np.array(color_list)
    if color_list.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if color_list.max() > 1.0:
        if color_list.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            color_list = color_list / 255.0
    return mpl.colors.LinearSegmentedColormap.from_list(name, color_list, 256)


def _register_cmaps():
    # make sure we are not trying to register color maps multiple times
    if hasattr(_register_cmaps, "called"):
        return
    for name, pal in _get_custom_palettes().items():
        mpl.colormaps.register(_list_to_colormap(pal, name))
        mpl.colormaps.register(_list_to_colormap(pal[::-1], f"{name}_r"))

    _register_cmaps.called = True


def _format_ticks(ax: plt.Axes, x: bool = True, y: bool = True, rotation: int = 45):
    """
    Function taken from https://cooltools.readthedocs.io/en/latest/notebooks/viz.html
    :param ax:        an Axes object.
    :param x:         if True, it formats labels in engineering notation for the x-axis
    :param y:         if True, it formats labels in engineering notation for the y-axis
    :param y:         if True, it formats labels in engineering notation for the y-axis
    :param rotation:  degrees of rotation of the x-axis ticks
    :return:          -
    """
    if x:
        ax.xaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.tick_bottom()

    if y:
        ax.yaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if rotation != 0:
        ax.tick_params(axis="x", rotation=rotation)


def hic_matrix(
    I: npt.NDArray,
    RoI: Tuple[int, int],
    title: Union[str, None] = None,
    cmap="fruit_punch",
    log_scale: bool = True,
    compact: bool = False,
    fig: Union[plt.Figure, None] = None,
    ax: Union[plt.Figure, None] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    :param I:                   Hi-C matrix to be plotted as image and saved
    :param  RoI:                refers to the Region of Interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                                (e.g., in genomic coordinates)
    :param title:               title to give to the image
    :param compact:          if False, it adds axes ticks, color bars
    :return:                    -
    """
    _register_cmaps()

    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    kwargs = {
        "extent": (RoI[0], RoI[1], RoI[1], RoI[0]),
        "cmap": cmap,
    }
    if log_scale:
        kwargs["norm"] = mpl.colors.LogNorm()
    else:
        kwargs["vmax"] = np.amax(I)

    ax.matshow(I, **kwargs)
    _format_ticks(ax)

    if compact:
        if title is not None:
            warnings.warn("value of the title parameter is ignored when compact is True")
        ax.axis("off")
    elif title is not None:
        fig.suptitle(title)

    ax.axis("scaled")

    return fig, ax
