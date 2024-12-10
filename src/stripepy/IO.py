# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT
import datetime
import json
import pathlib
import shutil
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import hictkpy
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter, ScalarFormatter

from stripepy.utils.stripe import Stripe

no_frills_in_images = False  # TODO safe removal of no_frills_in_images

fruit_punch = sns.blend_palette(["white", "red"], as_cmap=True)


class Result(object):
    def __init__(self, chrom: str):
        self._chrom = chrom
        self._roi = None
        self._min_persistence = None

        self._ut_all_minimum_points = None
        self._ut_all_maximum_points = None
        self._ut_persistence_of_all_minimum_points = None
        self._ut_persistence_of_all_maximum_points = None

        self._lt_all_minimum_points = None
        self._lt_all_maximum_points = None
        self._lt_persistence_of_all_minimum_points = None
        self._lt_persistence_of_all_maximum_points = None

        self._ut_persistent_minimum_points = None
        self._ut_persistent_maximum_points = None
        self._ut_persistence_of_minimum_points = None
        self._ut_persistence_of_maximum_points = None
        self._ut_pseudodistribution = None

        self._lt_persistent_minimum_points = None
        self._lt_persistent_maximum_points = None
        self._lt_persistence_of_minimum_points = None
        self._lt_persistence_of_maximum_points = None
        self._lt_pseudodistribution = None

        self._ut_stripes = None
        self._lt_stripes = None

    @property
    def _valid_attributes(self):
        return [a.removeprefix("_lt_") for a in dir(self) if a.startswith("_lt_")]

    @property
    def empty(self) -> bool:
        return self._lt_stripes is None and self._ut_stripes is None

    @property
    def chrom(self) -> str:
        return self._chrom

    @property
    def roi(self) -> Optional[Dict[str, List[int]]]:
        return self._roi

    @property
    def min_persistence(self) -> float:
        if self._min_persistence is None:
            raise RuntimeError('Attribute "min_persistence" is not set')

        return self._min_persistence

    def get(self, name: str, location: str) -> Union[List[Stripe], npt.NDArray[int], npt.NDArray[float]]:
        if location not in {"LT", "UT"}:
            raise RuntimeError("Location should be UT or LT")

        attr_name = f"_{location.lower()}_{name}"
        if not hasattr(self, attr_name):
            raise RuntimeError(
                f"No attribute named \"{name}\". Valid attributes are: {', '.join(self._valid_attributes)}"
            )

        attr = getattr(self, attr_name)
        if name == "stripes" and attr is None:
            return []

        if attr is None:
            raise RuntimeError(f'Attribute "{name}" is not set')

        return attr

    def get_stripes_descriptor(self, location: str, descriptor: str) -> Union[npt.NDArray[float], npt.NDArray[int]]:
        if not hasattr(Stripe, descriptor):
            raise RuntimeError(f'Stripe instance does not have an attribute named "{descriptor}"')

        stripes = self.get("stripes", location)

        if descriptor in {"seed", "left_bound", "right_bound", "top_bound", "bottom_bound"}:
            dtype = int
        else:
            dtype = float

        return np.array([getattr(stripe, descriptor) for stripe in stripes], dtype=dtype)

    def get_stripe_geo_descriptors(self, location: str) -> pd.DataFrame:
        descriptors = {
            "seed": "seed",
            "top_persistence": "seed persistence",
            "left_bound": "L-boundary",
            "right_bound": "R_boundary",
            "top_bound": "U-boundary",
            "bottom_bound": "D-boundary",
        }

        return pd.DataFrame(
            {name: self.get_stripes_descriptor(location, descriptor) for descriptor, name in descriptors.items()}
        )

    def get_stripe_bio_descriptors(self, location: str) -> pd.DataFrame:
        descriptors = {
            "inner_mean": "inner mean",
            "outer_mean": "outer mean",
            "rel_change": "relative change",
            "inner_std": "standard deviation",
        }

        return pd.DataFrame(
            {name: self.get_stripes_descriptor(location, descriptor) for descriptor, name in descriptors.items()}
        )

    def set_roi(self, coords: Dict[str, List[int]]):
        if self._roi is not None:
            raise RuntimeError("roi has already been set")

        self._roi = coords

    def set_min_persistence(self, min_persistence: float):
        if self._min_persistence is not None:
            raise RuntimeError("min_persistence has already been set")

        self._min_persistence = min_persistence

    def set(self, name: str, data: Union[Sequence[int], Sequence[float], Sequence[Stripe]], location: str):
        if location not in {"LT", "UT"}:
            raise RuntimeError("Location should be UT or LT")

        attr_name = f"_{location.lower()}_{name}"
        if not hasattr(self, attr_name):
            raise RuntimeError(
                f"No attribute named \"{name}\". Valid attributes are: {', '.join(self._valid_attributes)}"
            )

        if getattr(self, attr_name) is not None:
            raise RuntimeError(f'Attribute "{name}" for {location} has already been set')

        setattr(self, attr_name, np.array(data))


class ResultFile(object):
    def __init__(self, path: pathlib.Path, mode: str = "r"):
        self._path = path
        self._mode = mode

    def __enter__(self):
        self._h5 = h5py.File(self._path, self._mode)

        if self._mode == "r":
            self._validate(self._h5)

        self._version = self._h5.attrs.get("format-version", 1)
        self._attrs = dict(self._h5.attrs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._h5.close()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @staticmethod
    def _validate(h5: h5py.File):
        format = h5.attrs.get("format")  # noqa
        format_version = h5.attrs.get("format-version")
        try:
            if format is None:
                raise RuntimeError('attribute "format" is missing')

            if format_version is None:
                raise RuntimeError('attribute "format-version" is missing')

            if format != "HDF5::StripePy":
                raise RuntimeError(f'unrecognized file format: expected "HDF5::StripePy", found "{format}"')

            if format_version != 1:
                raise RuntimeError(
                    f'unsupported file format version "{format_version}". At present only version 1 is supported'
                )
        except RuntimeError as e:
            raise RuntimeError(
                f'failed to validate input file "{h5.filename}": {e}: file is corrupt or was not generated by StripePy.'
            )

    def init_file(self, matrix_file: hictkpy.File, normalization: str, metadata: Dict[str, Any]):
        self._h5.attrs["assembly"] = matrix_file.attributes().get("assembly", "unknown")
        self._h5.attrs["bin-size"] = matrix_file.resolution()
        self._h5.attrs["creation-date"] = datetime.datetime.now().isoformat()
        self._h5.attrs["format"] = "HDF5::StripePy"
        self._h5.attrs["format-url"] = "https://github.com/paulsengroup/StripePy"
        self._h5.attrs["format-version"] = 1
        self._h5.attrs["generated-by"] = f"StripePy v{version('stripepy')}"
        self._h5.attrs["metadata"] = json.dumps(metadata, indent=2)
        self._h5.attrs["normalization"] = normalization

        chroms = matrix_file.chromosomes(include_ALL=False)
        self._h5.create_group("/chroms")
        self._h5.create_dataset("/chroms/name", data=list(chroms.keys()))
        self._h5.create_dataset("/chroms/length", data=list(chroms.values()))

    def write_descriptors(self, result: Result):
        grp = self._h5.create_group(f"/{result.chrom}/global-pseudo-distribution/")

        grp.attrs.create("min_persistence_used", result.min_persistence)

        for location in ["UT", "LT"]:
            grp = self._h5.create_group(f"/{result.chrom}/global-pseudo-distribution/{location}")
            grp.create_dataset(
                "pseudo-distribution",
                data=result.get("pseudodistribution", location),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            grp.create_dataset(
                "minima_pts_and_persistence",
                data=np.array(
                    [
                        result.get("all_minimum_points", location),
                        result.get("persistence_of_all_minimum_points", location),
                    ]
                ),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            grp.create_dataset(
                "maxima_pts_and_persistence",
                data=np.array(
                    [
                        result.get("all_maximum_points", location),
                        result.get("persistence_of_all_maximum_points", location),
                    ]
                ),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            grp = self._h5.create_group(f"/{result.chrom}/stripes/{location}")
            descriptors = result.get_stripe_geo_descriptors(location)
            dset = grp.create_dataset(
                "geo-descriptors",
                data=descriptors.to_numpy(),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dset.attrs.create("col_names", data=descriptors.columns.tolist())

            descriptors = result.get_stripe_bio_descriptors(location)
            dset = grp.create_dataset(
                "bio-descriptors",
                data=descriptors.to_numpy(),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dset.attrs.create("col_names", data=descriptors.columns.tolist())


class ANSI:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


def remove_and_create_folder(path: pathlib.Path, force: bool):
    path = pathlib.Path(path)

    # Deleting folders:
    if path.exists():
        if not force:
            raise RuntimeError(f"output folder {path} already exists. Pass --force to overwrite it.")
        else:
            shutil.rmtree(path)

    # Create the folder:
    path.mkdir(parents=True)


def create_folders_for_plots(path: pathlib.Path):
    path = pathlib.Path(path)

    folders4plots = [
        path,
        path / "1_preprocessing",
        path / "2_TDA",
        path / "3_shape_analysis",
        path / "4_biological_analysis",
        path / "3_shape_analysis" / "local_pseudodistributions",
    ]

    # Creating folders:
    for folder2create in folders4plots:
        folder2create.mkdir(parents=True)

    return folders4plots


def format_ticks(ax, x=True, y=True, rotate=True):
    """
    Function taken from https://cooltools.readthedocs.io/en/latest/notebooks/viz.html
    :param ax:      an Axes object.
    :param x:       if True, it formats labels in engineering notation for the x-axis
    :param y:       if True, it formats labels in engineering notation for the y-axis
    :param y:       if True, it formats labels in engineering notation for the y-axis
    :param rotate:  if True, it rotates labels in the x-axis by 45 degrees.
    :return:        -
    """

    if x:
        ax.xaxis.set_major_formatter(EngFormatter("b"))
        ax.xaxis.tick_bottom()
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.tick_bottom()

    if y:
        ax.yaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if rotate:
        ax.tick_params(axis="x", rotation=45)


def HiC(I, RoI, plot_in_bp=False, output_folder=None, file_name=None, title=None, compactify=False):
    """
    :param I:                   Hi-C matrix to be plotted as image and saved
    :param  RoI:                refers to the Region of Interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                                (e.g., in genomic coordinates)
    :param plot_in_bp:          if True, labels are set in base pairs w.r.t. the genomic interval in RoI;
                                if False, labels are set in pixel coordinates
    :param output_folder:       path to folder where to save the image
    :param file_name:           name of the file to be created
    :param title:               title to give to the image
    :param compactify:          if False, it adds axes ticks, color bars
    :return:                    -
    """
    if output_folder is None or file_name is None:
        return

    fig, ax = plt.subplots(1, 1)
    img = ax.matshow(I, vmax=np.amax(I), extent=(RoI[0], RoI[1], RoI[3], RoI[2]), cmap=fruit_punch)
    if plot_in_bp:
        format_ticks(ax)

    if compactify is True:
        plt.axis("off")
    else:
        # plt.colorbar(img)
        if title is not None:
            fig.suptitle(title)

    fig.set_dpi(256)
    plt.axis("scaled")
    fig.tight_layout()

    plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    plt.close()


def pseudodistrib(
    pseudo_distrib,
    IoI,
    resolution,
    coords2scatter=None,
    colors=None,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    """
    :param pseudo_distrib:          1D ndarray representing a scalar function sampled over a uniform mesh
    :param  IoI:                    refers to the Interval of Interest [IoI[0], IoI[1]] (e.g., in genomic coordinates)
                                    where the scalar function was sampled on; see also plot_in_bp
    :param resolution:              resolution of the Hi-C matrix
    :param coords2scatter:          list of lists of genomic coordinates; each list of genomic coordinates determines
                                    a point cloud as follows: for genomic coordinate x, we sample the value of
                                    pseudo_distrib at x; each point cloud is scatterplotted with a specific color,
                                    potentially given in input, read below;
                                    if set to None, nothing happens
    :param colors:                  list of colors, one color per list of genomic coordinates (see coords2scatter);
                                    if set to None, use red
    :param output_folder:           path to folder where to save the image
    :param file_name:               name of the file to be created
    :param title:                   title to give to the image
    :param display:                 if False, it does not display the plot
    :return:                        -
    """

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        range(IoI[0], IoI[1], resolution),
        pseudo_distrib[int(IoI[0] / resolution) : int(IoI[1] / resolution)],
        color="red",
        linewidth=0.5,
        linestyle="solid",
    )
    if coords2scatter is not None:
        for n, cur_coords2scatter in enumerate(coords2scatter):
            if colors is not None:
                color = colors[n]
            else:
                color = "red"
            ax.plot(
                [cur_coord2scatter * resolution for cur_coord2scatter in cur_coords2scatter],
                pseudo_distrib[cur_coords2scatter],
                marker=".",
                linestyle="",
                markersize=6 * 1.5,
                color=color,
            )
    ax.xaxis.set_major_formatter(EngFormatter("b"))

    if no_frills_in_images is False:
        if title is not None:
            fig.suptitle(title)

    plt.ylim((0, 1))
    ax.set_xlabel("genomic coordinates (bp)")
    ax.set_ylabel("pseudo-distribution")
    fig.tight_layout()
    ax.grid(True)
    # plt.axis('scaled')

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display:
        plt.show()
    plt.close()


def pseudodistrib_and_HIoIs(
    pseudo_distrib, IoIs, resolution, colors=None, output_folder=None, file_name=None, title=None, display=False
):
    """
    :param pseudo_distrib:          1D ndarray representing a scalar function sampled over a uniform mesh
    :param  IoIs:                   list of lists, where the innermost lists are pairs of coordinates; the first pair
                                    refers to the Interval of Interest [IoI[0], IoI[1]] (e.g., in genomic coordinates)
                                    where pseudo-distribution is plotted; the remaining pairs define sub-regions to be
                                    plotted in (potentially) different colors
                                    where the scalar function was sampled on; see also plot_in_bp
    :param resolution:              resolution of the Hi-C matrix
    :param colors:                  list of colors, one color per pair of genomic coordinates (see IoIs);
                                    if set to None, use red
    :param output_folder:           path to folder where to save the image
    :param file_name:               name of the file to be created
    :param title:                   title to give to the image
    :param display:                 if False, it does not display the plot
    :return:                        -
    """

    fig, ax = plt.subplots(1, 1)
    for IoI, color in zip(IoIs, colors):
        ax.plot(
            range(IoI[0], IoI[1], resolution),
            pseudo_distrib[int(IoI[0] / resolution) : int(IoI[1] / resolution)],
            color=color,
            linewidth=0.5,
            linestyle="solid",
        )

    ax.xaxis.set_major_formatter(EngFormatter("b"))
    if no_frills_in_images is False:
        if title is not None:
            fig.suptitle(title)

    ax.set_xlabel("genomic coordinates (bp)")
    ax.set_ylabel("pseudo-distribution")
    fig.tight_layout()
    ax.grid(True)
    # plt.axis('scaled')

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display:
        plt.show()
    plt.close()


def HiC_and_sites(
    I,
    sites,
    RoI,
    resolution,
    where=None,
    plot_in_bp=False,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    """
    :param I:                  Hi-C matrix to be plotted as image and saved
    :param sites:              list of locations of interest
    :param  RoI:               refers to the region of interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                               (e.g., in genomic coordinates); see also plot_in_bp
    :param where:              if "lower" (resp. "upper"), then it plots sites only on the lower (resp. upper) part of
                               the Hi-C matrix; otherwise, it plots sites spanning the whole Hi-C matrix
    :param plot_in_bp:         if True, labels are set in base pairs
    :param output_folder:      path to folder where to save the image
    :param file_name:          name of the file to be created
    :param title:              title to give to the image
    :param display:            if False, it does not display the plot
    :return:                   -
    """

    fig, ax = plt.subplots(1, 1)
    img = ax.matshow(I, vmax=np.amax(I), extent=(RoI[0], RoI[1], RoI[3], RoI[2]), cmap=fruit_punch)
    if plot_in_bp:
        format_ticks(ax)

    for site in [site * resolution for site in sites]:
        if where == "lower":
            ax.plot(
                [site, site], [site, RoI[1] - 1 * resolution], color=(0.0, 0.0, 1.0), linestyle="dashed", linewidth=1
            )
        elif where == "upper":
            ax.plot(
                [site, site], [RoI[0] + 1 * resolution, site], color=(0.0, 0.0, 1.0), linestyle="dashed", linewidth=1
            )
        else:
            ax.plot(
                [site, site],
                [RoI[0] + 1 * resolution, RoI[2] - 1 * resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )

    if no_frills_in_images is False:
        plt.colorbar(img)
        if title is not None:
            fig.suptitle(title)
    else:
        plt.axis("off")

    fig.set_dpi(256)
    plt.axis("scaled")
    fig.tight_layout()

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display is True:
        plt.show()
    plt.close()


def HiC_and_HIoIs(
    I,
    HIoIs,
    RoI,
    resolution,
    where=None,
    plot_in_bp=False,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    """
    :param I:                   Hi-C matrix to be plotted as image and saved
    :param HIoIs:               list of lists, where the innermost lists are pairs of elements
    :param  RoI:                refers to the region of interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                                (e.g., in genomic coordinates); see also plot_in_bp
    :param resolution:              resolution of the Hi-C matrix
    :param where:               if "lower" (resp. "upper"), then it plots sites only on the lower (resp. upper) part of
                                the Hi-C matrix; otherwise, it plots sites spanning the whole Hi-C matrix
    :param plot_in_bp:          if True, labels are set in base pairs
    :param output_folder:       path to folder where to save the image
    :param file_name:           name of the file to be created
    :param title:               title to give to the image
    :param display:             if False, it does not display the plot
    :return:                    -
    """

    fig, ax = plt.subplots(1, 1)
    img = ax.matshow(I, vmax=np.amax(I), extent=(RoI[0], RoI[1], RoI[3], RoI[2]), cmap=fruit_punch)
    if plot_in_bp:
        format_ticks(ax)

    for HIoI in HIoIs:
        if where == "lower":
            ax.plot(
                [RoI[0] + HIoI[0], RoI[0] + HIoI[0]],
                [RoI[2] + HIoI[0], RoI[2] + (I.shape[0] - 1) * resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + HIoI[1] + resolution, RoI[0] + HIoI[1] + resolution],
                [RoI[2] + HIoI[1], RoI[2] + (I.shape[0] - 1) * resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )
        elif where == "upper":
            ax.plot(
                [RoI[0] + HIoI[0], RoI[0] + HIoI[0]],
                [RoI[2] + 1 * resolution, RoI[2] + HIoI[0]],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + HIoI[1] + resolution, RoI[0] + HIoI[1] + resolution],
                [RoI[2] + 1 * resolution, RoI[2] + HIoI[1] + resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )
        else:
            ax.plot(
                [RoI[0] + HIoI[0], RoI[0] + HIoI[0]],
                [RoI[2] + 1 * resolution, RoI[2] + (I.shape[0] - 1) * resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + HIoI[1] + resolution, RoI[0] + HIoI[1] + resolution],
                [RoI[2] + 1 * resolution, RoI[2] + (I.shape[0] - 1) * resolution],
                color=(0.0, 0.0, 1.0),
                linestyle="dashed",
                linewidth=1,
            )

    if no_frills_in_images is False:
        plt.colorbar(img)
        if title is not None:
            fig.suptitle(title)
    else:
        plt.axis("off")

    fig.set_dpi(256)
    plt.axis("scaled")
    fig.tight_layout()

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display:
        plt.show()
    plt.close()


def plot_stripes(
    I,
    LT_HIoIs,
    LT_VIoIs,
    UT_HIoIs,
    UT_VIoIs,
    RoI,
    resolution,
    plot_in_bp=False,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    """
    :param I:                  Hi-C matrix to be plotted as image and saved
    :param LT_HIoIs:           Horizontal Intervals of Interest (lower-triangular part)
    :param LT_VIoIs:           Vertical Intervals of Interest (lower-triangular part)
    :param UT_HIoIs:           Horizontal Intervals of Interest (upper-triangular part)
    :param UT_VIoIs:           Vertical Intervals of Interest (upper-triangular part)
    :param RoI:                refers to the region of interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                               (e.g., in genomic coordinates); see also plot_in_bp
    :param plot_in_bp:         if True, labels are set in base pairs
    :param output_folder:      path to folder where to save the image
    :param file_name:          name of the file to be created
    :param title:              title to give to the image
    :param display:            if False, it does not display the plot
    :return:                   -
    """

    fig, ax = plt.subplots(1, 1)
    ax.matshow(I, vmax=np.amax(I), extent=(RoI[0], RoI[1], RoI[3], RoI[2]), cmap=fruit_punch)
    if plot_in_bp:
        format_ticks(ax)

    # Upper-triangular candidates:
    for num_window, (HIoI, VIoI) in enumerate(zip(UT_HIoIs, UT_VIoIs)):
        low_idx_up_opt = max(1, VIoI[0]) * resolution
        low_idx_dw_opt = min(VIoI[1], I.shape[0] - 1) * resolution
        low_idx_sx_opt = max(1, HIoI[0]) * resolution
        low_idx_dx_opt = min(HIoI[1], I.shape[1] - 1) * resolution
        if low_idx_up_opt != low_idx_dw_opt and low_idx_sx_opt != low_idx_dx_opt:
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_sx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_dx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_dw_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_up_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            # plt.text(RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_dw_opt + 15, str(num_window))

    for num_window, (HIoI, VIoI) in enumerate(zip(LT_HIoIs, LT_VIoIs)):
        upp_idx_up_opt = max(1, VIoI[0]) * resolution
        upp_idx_dw_opt = min(VIoI[1], I.shape[0] - 1) * resolution
        upp_idx_sx_opt = max(1, HIoI[0]) * resolution
        upp_idx_dx_opt = min(HIoI[1], I.shape[0] - 1) * resolution
        if upp_idx_up_opt != upp_idx_dw_opt and upp_idx_sx_opt != upp_idx_dx_opt:
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_sx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_dx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_dw_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_up_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            # plt.text(RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_up_opt - 15, str(num_window))

    if no_frills_in_images is False:
        if title is not None:
            fig.suptitle(title)
    else:
        plt.axis("off")

    fig.set_dpi(256)
    fig.tight_layout()

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display:
        plt.show()
    plt.close()


def plot_stripes_and_peaks(
    I,
    LT_HIoIs,
    LT_VIoIs,
    UT_HIoIs,
    UT_VIoIs,
    LT_peaks_ids,
    UT_peaks_ids,
    RoI,
    resolution,
    plot_in_bp=False,
    output_folder=None,
    file_name=None,
    title=None,
    display=False,
):
    # TODO CORREGGERE LA TRASLAZIONE DI ROI
    """
    :param I:                  Hi-C matrix to be plotted as image and saved
    :param LT_HIoIs:           Horizontal Intervals of Interest (lower-triangular part)
    :param LT_VIoIs:           Vertical Intervals of Interest (lower-triangular part)
    :param UT_HIoIs:           Horizontal Intervals of Interest (upper-triangular part)
    :param UT_VIoIs:           Vertical Intervals of Interest (upper-triangular part)
    :param RoI:                refers to the region of interest [RoI[0], RoI[1]]x[RoI[2], RoI[3]]
                               (e.g., in genomic coordinates); see also plot_in_bp
    :param plot_in_bp:         if True, labels are set in base pairs
    :param output_folder:      path to folder where to save the image
    :param file_name:          name of the file to be created
    :param title:              title to give to the image
    :param display:            if False, it does not display the plot
    :return:                   -
    """

    fig, ax = plt.subplots(1, 1)
    ax.matshow(I, vmax=np.amax(I), extent=(RoI[0], RoI[1], RoI[3], RoI[2]), cmap=fruit_punch)
    if plot_in_bp:
        format_ticks(ax)

    # Upper-triangular candidates:
    for num_window, (HIoI, VIoI, peaks_ids) in enumerate(zip(UT_HIoIs, UT_VIoIs, UT_peaks_ids)):
        low_idx_up_opt = max(1, VIoI[0]) * resolution
        low_idx_dw_opt = min(VIoI[1], I.shape[0] - 1) * resolution
        low_idx_sx_opt = max(1, HIoI[0]) * resolution
        low_idx_dx_opt = min(HIoI[1], I.shape[1] - 1) * resolution
        if low_idx_up_opt != low_idx_dw_opt and low_idx_sx_opt != low_idx_dx_opt:
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_sx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_dx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_dw_opt, RoI[2] + low_idx_dw_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + low_idx_sx_opt, RoI[2] + low_idx_dx_opt],
                [RoI[0] + low_idx_up_opt, RoI[2] + low_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )

            transl_peak_roots_ids = [RoI[0] + (low_idx_sx_opt + low_idx_dx_opt) / 2 for _ in peaks_ids]
            transl_peaks_ids = [peak_idx * resolution + RoI[2] for peak_idx in peaks_ids]
            ax.scatter(transl_peak_roots_ids, transl_peaks_ids, c="blue")

    for num_window, (HIoI, VIoI, peaks_ids) in enumerate(zip(LT_HIoIs, LT_VIoIs, LT_peaks_ids)):
        upp_idx_up_opt = max(1, VIoI[0]) * resolution
        upp_idx_dw_opt = min(VIoI[1], I.shape[0] - 1) * resolution
        upp_idx_sx_opt = max(1, HIoI[0]) * resolution
        upp_idx_dx_opt = min(HIoI[1], I.shape[0] - 1) * resolution
        if upp_idx_up_opt != upp_idx_dw_opt and upp_idx_sx_opt != upp_idx_dx_opt:
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_sx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_dx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_dw_opt, RoI[2] + upp_idx_dw_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )
            ax.plot(
                [RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_dx_opt],
                [RoI[0] + upp_idx_up_opt, RoI[2] + upp_idx_up_opt],
                "blue",
                linestyle="dashed",
                linewidth=1,
            )

            transl_peak_roots_ids = [RoI[0] + (upp_idx_sx_opt + upp_idx_dx_opt) / 2 for _ in peaks_ids]
            transl_peaks_ids = [peak_idx * resolution + RoI[2] for peak_idx in peaks_ids]
            ax.scatter(transl_peak_roots_ids, transl_peaks_ids, c="blue")
            # plt.text(RoI[0] + upp_idx_sx_opt, RoI[2] + upp_idx_up_opt - 15, str(num_window))

    if no_frills_in_images is False:
        if title is not None:
            fig.suptitle(title)
    else:
        plt.axis("off")

    fig.set_dpi(256)
    fig.tight_layout()

    if output_folder is not None and file_name is not None:
        plt.savefig(f"{output_folder}/{file_name}", bbox_inches="tight")

    if display:
        plt.show()
    plt.close()


def save_candidates_bedpe(HIoIs, VIoIs, resolution, chr, output_folder, file_name):
    """
    :param HIoIs:              Horizontal Intervals of Interest
    :param VIoIs:              Vertical Intervals of Interest
    :param resolution:         resolution
    :param chr:                chromosome
    :param output_folder:      path to folder where to save the image
    :param file_name:          name of the file to be created
    :return:                   -
    """

    with open(f"{output_folder}/{file_name}", "w") as f:
        for HIoI, VIoI in zip(HIoIs, VIoIs):
            f.write(
                f"{chr}\t{resolution * HIoI[0]}\t{resolution * HIoI[1]}\t"
                f"{chr}\t{resolution * VIoI[0]}\t{resolution * VIoI[1]}\n"
            )
