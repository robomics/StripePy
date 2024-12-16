# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import warnings
from typing import Dict, List, Optional, Tuple

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


def _find_seeds_in_RoI(
    seeds: List[int], left_bound_RoI: int, right_bound_RoI: int
) -> Tuple[npt.NDArray[int], List[int]]:
    """
    Select seed coordinates that fall within the given left and right boundaries.

    Parameters
    ----------
    seeds: List[int]
        a list with the seed coordinates
    left_bound_RoI: int
        left bound of the region of interest
    right_bound_RoI: int
        right bound of the region of interest

    Returns
    -------
    Tuple[NDArray[int], List[int]]
        a tuple consisting of:

         * the indices of seed coordinates falling within the given boundaries
         * the coordinates of the selected seeds
    """

    assert left_bound_RoI >= 0
    assert right_bound_RoI >= left_bound_RoI

    # Find sites within the range of interest -- lower-triangular:
    ids_seeds_in_RoI = np.where((left_bound_RoI <= np.array(seeds)) & (np.array(seeds) <= right_bound_RoI))[0]
    seeds_in_RoI = np.array(seeds)[ids_seeds_in_RoI].tolist()

    return ids_seeds_in_RoI, seeds_in_RoI


def hic_matrix(
    I: npt.NDArray,
    RoI: Tuple[int, int],
    title: Optional[str] = None,
    cmap="fruit_punch",
    log_scale: bool = True,
    compact: bool = False,
    with_colorbar: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, AxesImage]:
    """
    :param I:                   Hi-C matrix to be plotted as image and saved
    :param  RoI:                refers to the Region of Interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                                (e.g., in genomic coordinates)
    :param title:               title to give to the image
    :param compact:             if False, it adds axes ticks, color bars
    :param fig:                 figure to use for plotting
    :param ax:                  axis to use for plotting
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

    img = ax.imshow(I, **kwargs)
    _format_ticks(ax)

    ax.axis("scaled")

    if compact:
        ax.axis("off")

    if title is not None:
        if compact:
            warnings.warn("value of the title parameter is ignored when compact is True")
        else:
            fig.suptitle(title)

    if with_colorbar:
        if compact:
            warnings.warn("the with_colorbar parameter is ignored when compact is True")
        else:
            fig.colorbar(img, ax=ax)

    return fig, ax, img


def pseudodistribution(
    pseudo_distrib_lt: npt.NDArray[float],
    pseudo_distrib_ut: npt.NDArray[float],
    IoI: Tuple[int, int],
    resolution: int,
    title: Optional[str] = None,
    coords2scatter_lt: Optional[npt.NDArray[int]] = None,
    coords2scatter_ut: Optional[npt.NDArray[int]] = None,
    colors: Optional[List] = None,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    :param pseudo_distrib_lt:       1D ndarray representing a scalar function sampled over a uniform mesh (lower-triangular matrix)
    :param pseudo_distrib_ut:       same as pseudo_distrib_lt but for the upper-triangular matrix
    :param IoI:                     refers to the Interval of Interest [IoI[0], IoI[1]] (e.g., in genomic coordinates)
                                    where the scalar function was sampled on; see also plot_in_bp
    :param resolution:              resolution of the Hi-C matrix
    :param title:                   title to give to the image
    :param coords2scatter_lt:       list of lists of genomic coordinates (lower-triangular matrix);
                                    each list of genomic coordinates determines a point cloud as follows: for genomic coordinate x, we sample the value of
                                    pseudo_distrib at x; each point cloud is scatterplotted with a specific color,
                                    potentially given in input, read below;
                                    if set to None, nothing happens
    :param coords2scatter_ut:       same as coords2scatter_lt but for the upper-triangular matrix
    :param colors:                  list of colors, one color per list of genomic coordinates (see coords2scatter);
                                    if set to None, use red
    :param fig:                     figure to use for plotting
    :param axs:                     a tuple of two axes to use for plotting
    """
    if fig is None:
        if axs is not None:
            raise RuntimeError("axs should be None when fig is None")
        fig, axs = plt.subplots(2, 1, sharex=True)
    elif axs is None:
        raise RuntimeError("axs cannot be None when fig is not None")

    if len(axs) != 2:
        raise RuntimeError("axs should be a tuple with exactly two plt.Axes")

    i1 = IoI[0] // resolution
    i2 = IoI[1] // resolution

    def _plot(data, coords2scatter, ax, title):
        ax.plot(
            np.arange(i1, i2) * resolution,
            data[i1:i2],
            color="red",
            linewidth=0.5,
            linestyle="solid",
        )
        if coords2scatter is not None:
            for n, cur_coords2scatter in enumerate(coords2scatter):
                if colors is not None:
                    color = colors[n]
                else:
                    color = "blue"
                ax.plot(
                    np.array(cur_coords2scatter) * resolution,
                    data[cur_coords2scatter],
                    marker=".",
                    linestyle="",
                    markersize=5,
                    color=color,
                )
        ax.set(title=title, ylim=(0, 1), ylabel="Pseudo-distribution")
        ax.grid(True)

    _plot(pseudo_distrib_ut, coords2scatter_ut, axs[0], "Upper Triangular")
    _plot(pseudo_distrib_lt, coords2scatter_lt, axs[1], "Lower Triangular")

    _format_ticks(axs[1], y=False, x=True)

    axs[1].set(xlabel="Genomic coordinates (bp)")

    if title is not None:
        fig.suptitle(title)

    return fig, axs


def plot_sites(
    sites: npt.NDArray[int],
    RoI: Tuple[int, int],
    location: Optional[str],
    color: str = "blue",
    title: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    :param sites:              list of locations of interest. Locations should be expressed in genomic coordinates
    :param RoI:                refers to the region of interest RoI[0], RoI[1] in genomic coordinates
    :param location:           if "lower" (resp. "upper"), then it plots sites only on the lower (resp. upper) part of
                               the Hi-C matrix; otherwise, it plots sites spanning the whole Hi-C matrix
    :param color:              color used for plotting
    :param title:              title to give to the image
    :param fig:                figure to use for plotting
    :param ax:                 axis to use for plotting
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
        x = [site] * 2
        if location == "lower":
            y = [site, RoI[1] - 1]
        elif location == "upper":
            y = [RoI[0] + 1, site]
        else:
            y = [RoI[0] + 1, RoI[1] - 1]

        if site < RoI[0] or site > RoI[1]:
            raise RuntimeError("site outside of bounds")

        ax.plot(x, y, color=color, linestyle="dashed", linewidth=1)

    _format_ticks(ax)

    if title is not None:
        fig.suptitle(title)

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
