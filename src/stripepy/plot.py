# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage
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


def _format_ticks(ax: plt.Axes, xaxis: bool = True, yaxis: bool = True, rotation: int = 30):
    """
    Function taken from https://cooltools.readthedocs.io/en/latest/notebooks/viz.html
    """
    if xaxis:
        ax.xaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.tick_bottom()

    if yaxis:
        ax.yaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if rotation != 0:
        ax.tick_params(axis="x", rotation=rotation)


def hic_matrix(
    matrix: npt.NDArray,
    region: Tuple[int, int],
    cmap="fruit_punch",
    log_scale: bool = True,
    with_colorbar: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, AxesImage]:
    """
    Plot the given matrix as a heatmap

    Parameters
    ----------
    matrix: npt.NDArray
        the 2D matrix to be plotted
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    cmap
        color map to be used for plotting
    log_scale: bool
        if True, plot the heatmap in log scale
    with_colorbar: bool
        if True, add a color bar for the plotted heatmap
    fig: Optional[plt.Figure]
        figure to use for plotting
    ax: Optional[plt.Axes]
        axis to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax: plt.Axes
        axis used for plotting
    img:
        image returned by ax.imshow()
    """
    _register_cmaps()

    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    kwargs = {
        "extent": (region[0], region[1], region[1], region[0]),
        "cmap": cmap,
    }
    if log_scale:
        kwargs["norm"] = mpl.colors.LogNorm()
    else:
        kwargs["vmax"] = np.amax(matrix)

    img = ax.imshow(matrix, **kwargs)
    _format_ticks(ax)

    if with_colorbar:
        fig.colorbar(img, ax=ax)

    return fig, ax, img


def pseudodistribution(
    pseudo_distrib_lt: Sequence[float],
    pseudo_distrib_ut: Sequence[float],
    region: Tuple[int, int],
    resolution: int,
    highlighted_points_lt: Optional[Sequence[int]] = None,
    highlighted_points_ut: Optional[Sequence[int]] = None,
    colors: Optional[Any] = None,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot the given pseudo-distributions as a two line plots

    Parameters
    ----------
    pseudo_distrib_lt: Sequence[float]
        pseudo-distribution values computed on the lower-triangular matrix
    pseudo_distrib_ut: Sequence[float]
        pseudo-distribution values computed on the upper-triangular matrix
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    resolution: int
        resolution (bp) of the Hi-C matrix from which the pseudo-distribution were computed
    highlighted_points_lt: Optional[Sequence[int]]
        list of genomic coordinates (lower-triangular matrix) to be highlighted.
        Each pair of genomic coordinates determines a point cloud as follows:
        - for genomic coordinate x, we sample the value of pseudo_distrib_lt at x
        - each point cloud is scatterplotted with the color specified through the colors parameter.
    highlighted_points_ut: Optional[Sequence[int]]
        same as coords2scatter_lt but for the upper-triangular matrix
    colors: Optional[Any]
        one or more colors to be used when generating the scatter plot.
        When provided it should be either:
        - a single color: all points are plotted using the same color
        - a list colors: the list should contain at least
          max(len(color2scatter_lt), len(color2scatter_ut)) colors
    fig: Optional[plt.Figure]
        figure to use for plotting
    axs: Optional[Tuple[plt.Axes, plt.Axes]]
        axes to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax1: plt.Axes
        axis with the plot for upper-triangular pseudo-distribution
    ax2: plt.Axes
        axis with the plot for lower-triangular pseudo-distribution
    """
    if fig is None:
        if axs is not None:
            raise RuntimeError("axs should be None when fig is None")
        fig, axs = plt.subplots(2, 1, sharex=True)
    elif axs is None:
        raise RuntimeError("axs cannot be None when fig is not None")

    if len(axs) != 2:
        raise RuntimeError("axs should be a tuple with exactly two plt.Axes")

    i1 = region[0] // resolution
    i2 = region[1] // resolution

    if colors is None:
        num_colors = 0
        if highlighted_points_lt is not None:
            num_colors = len(highlighted_points_lt)
        if highlighted_points_ut is not None:
            num_colors = max(num_colors, len(highlighted_points_ut))

        if num_colors > 0:
            colors = ["blue"] * num_colors

    def _plot(data, points, ax, title):
        ax.plot(
            np.arange(i1, i2) * resolution,
            data[i1:i2],
            color="red",
            linewidth=0.5,
            linestyle="solid",
        )
        if points is not None:
            for n, x in enumerate(points):
                if not region[0] <= x <= region[1]:
                    raise RuntimeError("point outside of bounds")
                ax.plot(
                    x,
                    data[x // resolution],
                    marker=".",
                    linestyle="",
                    markersize=5,
                    color=colors[n],
                )
        ax.set(title=title, ylim=(0, 1), ylabel="Pseudo-distribution")
        ax.grid(True)

    _plot(pseudo_distrib_ut, highlighted_points_ut, axs[0], "Upper Triangular")
    _plot(pseudo_distrib_lt, highlighted_points_lt, axs[1], "Lower Triangular")

    _format_ticks(axs[1], yaxis=False, xaxis=True)

    axs[1].set(xlabel="Genomic coordinates (bp)")

    return fig, axs


def plot_sites(
    sites: Sequence[int],
    region: Tuple[int, int],
    location: Optional[str],
    color="blue",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot one or more sites of interest (e.g. stipe seeds).

    Parameters
    ----------
    sites: Sequence[int]
        list of genomic coordinates (bp) corresponding to the x-coordinates of the sites to be plotted
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    location: Optional[str]
        if "lower" (resp. "upper"), then plot sites only on the lower (resp. upper) part of
        the Hi-C matrix. Otherwise, plot sites spanning the whole Hi-C matrix
    color:
        color to be used for plotting
    fig: Optional[plt.Figure]
        figure to use for plotting
    ax: Optional[Tuple[plt.Axes, plt.Axes]]
        axis to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax: plt.Axes
        axis used for plotting
    """
    if location is not None and location not in {"lower", "upper"}:
        raise ValueError('location should be "lower" or "upper"')

    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    if len(sites) == 0:
        warnings.warn("stripepy.plot.plot_sites: no sites to plot!")

    for site in sites:
        if not region[0] <= site <= region[1]:
            raise RuntimeError("site outside of bounds")

        x = [site] * 2
        if location == "lower":
            y = [site, region[1] - 1]
        elif location == "upper":
            y = [region[0] + 1, site]
        else:
            y = [region[0] + 1, region[1] - 1]

        ax.plot(x, y, color=color, linestyle="dashed", linewidth=1)

    _format_ticks(ax)

    return fig, ax


def mask_regions_1d(
    matrix: npt.NDArray,
    resolution: int,
    location: Optional[str],
    whitelist: Optional[List[Tuple[int, int]]] = None,
    blacklist: Optional[List[Tuple[int, int]]] = None,
) -> npt.NDArray:
    """
    Mask rows or columns of the given matrix based on whitelisted or blacklisted regions.

    Parameters
    ----------
    matrix: npt.NDArray
        matrix to be masked. The matrix is expected to be a 2D matrix and be symmetric.
    resolution: int
        matrix resolution in bp.
    location: str
        location where the selective masking should be applied.
        When "lower", values in the upper triangular matrix are all set to 0 and values
        in the lower triangular matrix are masked based on whitelisted or blacklisted regions.
        When "upper", values in the lower triangular matrix are all set to 0 and values
        in the upper triangular matrix are masked based on whitelisted or blacklisted regions.
    whitelist: Optional[List[Tuple[int, int]]]
        list of regions to NOT be masked. Regions should be expressed in matrix coordinates.
    blacklist: Optional[List[Tuple[int, int]]]
        list of regions to be masked. Regions should be expressed in matrix coordinates.

    Returns
    -------
    matrix: npt.NDArray
        the masked matrix
    """
    if location is not None and location not in {"lower", "upper"}:
        raise ValueError('location should be "lower" or "upper"')

    if (whitelist is None and blacklist is None) or (whitelist is not None and blacklist is not None):
        raise ValueError("please specify either whitelist or blacklist")

    m = matrix.copy()
    mask = None
    if whitelist is not None:
        idx = []
        if len(whitelist) > 0:
            for i1, i2 in whitelist:
                i1 //= resolution
                i2 //= resolution
                idx.extend(list(range(i1, i2 + 1)))

        mask = np.setdiff1d(np.arange(m.shape[0]), np.unique(idx))

    if blacklist is not None:
        mask = []
        if len(blacklist) > 0:
            for i1, i2 in blacklist:
                i1 //= resolution
                i2 //= resolution
                mask.extend(list(range(i1, i2 + 1)))

        mask = np.unique(mask)

    if mask is not None:
        m[:, mask] = 0

    if location == "upper":
        return np.triu(m)
    if location == "lower":
        return np.tril(m)
    return np.tril(m.T) + np.triu(m)


def draw_boxes(
    regions: List[Tuple[int, int, int, int]],
    bound_box: Tuple[int, int],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw bounding boxes around a list of regions of interest.

    Parameters
    ----------
    regions:
        a list of genomic regions, each consisting of a 4-element tuple (x, y, width, height).
        All coordinates are in genomic units (i.e. bp).
    bound_box:
        first and last genomic position that should be plotted.
        Used to ensure that boxes do not extend outside the plotted area.
    fig:
        figure to use for plotting
    ax:
        axis to use for plotting
    **kwargs:
        additional keyword arguments to pass to plt.plot()

    Returns
    -------
    figure, axes:
        the plt.Figure and plt.Axes used for plotting
    """
    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "dashed"

    for x, y, width, height in regions:
        x = np.clip([x, x, x + width, x + width, x], bound_box[0], bound_box[1])
        y = np.clip([y, y + height, y + height, y + width, y], bound_box[0], bound_box[1])
        ax.plot(x, y, **kwargs)
    return fig, ax
