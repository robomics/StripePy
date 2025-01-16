# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import json
import multiprocessing as mp
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import structlog

from stripepy import IO, others, stripepy
from stripepy.utils.common import _import_matplotlib, pretty_format_elapsed_time
from stripepy.utils.progress_bar import initialize_progress_bar


def _generate_metadata_attribute(configs_input: Dict[str, Any], configs_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "constrain-heights": configs_thresholds["constrain_heights"],
        "genomic-belt": configs_input["genomic_belt"],
        "global-persistence-minimum": configs_thresholds["glob_pers_min"],
        "local-persistence-minimum": configs_thresholds["loc_pers_min"],
        "local-trend-minimum": configs_thresholds["loc_trend_min"],
        "max-width": configs_thresholds["max_width"],
        "min-chromosome-size": configs_thresholds["min_chrom_size"],
    }


def _plan(chromosomes: Dict[str, int], min_size: int, logger=None) -> List[Tuple[str, int, bool]]:
    plan = []
    small_chromosomes = []
    for chrom, length in chromosomes.items():
        skip = length <= min_size
        plan.append((chrom, length, skip))
        if skip:
            small_chromosomes.append(chrom)

    if len(small_chromosomes) != 0:
        if logger is None:
            logger = structlog.get_logger()

        logger.warning(
            "the following chromosomes are discarded because shorter than --min-chrom-size=%d bp: %s",
            min_size,
            ", ".join(small_chromosomes),
        )

    return plan


def _generate_empty_result(chrom: str, chrom_size: int, resolution: int) -> IO.Result:
    result = IO.Result(chrom, chrom_size)
    result.set_min_persistence(0)

    num_bins = (chrom_size + resolution - 1) // resolution
    for location in ("LT", "UT"):
        result.set("all_minimum_points", [], location)
        result.set("all_maximum_points", [], location)
        result.set("persistence_of_all_minimum_points", [], location)
        result.set("persistence_of_all_maximum_points", [], location)
        result.set("persistent_minimum_points", [], location)
        result.set("persistent_maximum_points", [], location)
        result.set("persistence_of_minimum_points", [], location)
        result.set("persistence_of_maximum_points", [], location)
        result.set("pseudodistribution", np.full(num_bins, np.nan, dtype=float), location)
        result.set("stripes", [], location)

    return result


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, pathlib.Path):
            return str(o)
        return super().default(o)


def _write_param_summary(*configs: Dict[str, Any]):
    config = {}
    for c in configs:
        config.update(c)

    config_str = json.dumps(config, indent=2, sort_keys=True, cls=_JSONEncoder)
    structlog.get_logger().info(f"CONFIG:\n{config_str}")


def _compute_progress_bar_weights(chrom_sizes: Dict[str, int], include_plotting: bool, nproc: int) -> pd.DataFrame:
    # These weights have been computed on a Linux machine (Ryzen 9 7950X3D) using 1 core to process
    # 4DNFI9GMP2J8.mcool at 10kbp
    step_weights = {
        "input": 0.035557,
        "step_1": 0.018159,
        "step_2": 0.015722,
        "step_3": 0.301879,
        "step_4": 0.051055,
        "output": 0.054285,
        "step_5": 0.523344 / nproc,
    }

    if not include_plotting:
        step_weights["step_5"] = 0.0

    tot = sum(step_weights.values())
    step_weights = {k: v / tot for k, v in step_weights.items()}

    weights = []
    for size in chrom_sizes.values():
        weights.extend((size * w for w in step_weights.values()))

    weights = np.array(weights)
    weights /= weights.sum()
    weights = np.minimum(weights.cumsum(), 1.0)

    shape = (len(chrom_sizes), len(step_weights))
    df = pd.DataFrame(weights.reshape(shape), columns=list(step_weights.keys()))
    df["chrom"] = list(chrom_sizes.keys())
    return df.set_index(["chrom"])


def _init_mpl_backend(skip: bool):
    if skip:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        structlog.get_logger().warning("failed to initialize matplotlib backend")
        pass


def run(
    configs_input: Dict[str, Any],
    configs_thresholds: Dict[str, Any],
    configs_output: Dict[str, Any],
    configs_other: Dict[str, Any],
) -> int:
    # How long does stripepy take to analyze the whole Hi-C matrix?
    start_global_time = time.time()

    _write_param_summary(configs_input, configs_thresholds, configs_output, configs_other)

    if configs_input["roi"] is not None:
        # Raise an error immediately if --roi was passed and matplotlib is not available
        _import_matplotlib()

    # Data loading:
    f = others.open_matrix_file_checked(configs_input["contact_map"], configs_input["resolution"])

    # Remove existing folders:
    # configs_output["output_folder"] = (
    #     f"{configs_output['output_folder']}/{configs_input['contact_map'].stem}/{configs_input['resolution']}"
    # )
    IO.remove_and_create_folder(configs_output["output_folder"], configs_output["force"])

    with contextlib.ExitStack() as ctx:
        main_logger = structlog.get_logger()

        # Create HDF5 file to store candidate stripes:
        main_logger.info('initializing result file "%s"...', configs_output["output_folder"] / "results.hdf5")
        h5 = ctx.enter_context(IO.ResultFile(configs_output["output_folder"] / "results.hdf5", "w"))

        h5.init_file(f, configs_input["normalization"], _generate_metadata_attribute(configs_input, configs_thresholds))

        # Set up the process pool when appropriate
        if configs_other["nproc"] > 1:
            main_logger.debug("initializing a pool of %d processes...", configs_other["nproc"])
            pool = ctx.enter_context(
                mp.Pool(
                    processes=configs_other["nproc"],
                    initializer=_init_mpl_backend,
                    initargs=(configs_input["roi"] is None,),
                )
            )
        else:
            pool = None

        disable_bar = not sys.stderr.isatty()
        progress_weights_df = _compute_progress_bar_weights(
            f.chromosomes(), include_plotting=configs_input["roi"] is not None, nproc=configs_other["nproc"]
        )
        progress_bar = ctx.enter_context(
            initialize_progress_bar(
                total=sum(f.chromosomes().values()),
                manual=True,
                disable=disable_bar,
                enrich_print=False,
                file=sys.stderr,
                receipt=False,
                refresh_secs=0.05,
                monitor="{percent:.2%}",
                unit="bp",
                scale="SI",
            )
        )

        # Lopping over all chromosomes:
        for chrom_name, chrom_size, skip in _plan(
            f.chromosomes(include_ALL=False), configs_thresholds["min_chrom_size"]
        ):
            progress_weights = progress_weights_df.loc[chrom_name, :].to_dict()

            if skip:
                logger.warning("writing an empty entry for chromosome %s...", chrom_name)
                result = _generate_empty_result(chrom_name, chrom_size, configs_input["resolution"])
                h5.write_descriptors(result)
                progress_bar(max(progress_weights.values()))
                continue

            logger = main_logger.bind(chrom=chrom_name)
            logger.info("begin processing...")
            start_local_time = time.time()

            # Removing and creating folders to store output files:
            # configs_input['roi'] = None
            if configs_input["roi"] is not None:
                IO.create_folders_for_plots(configs_output["output_folder"] / "plots" / chrom_name)

            logger.debug("fetching interactions using normalization=%s", configs_input["normalization"])
            I = f.fetch(chrom_name, normalization=configs_input["normalization"]).to_csr("full")
            progress_bar(progress_weights["input"])

            # RoI:
            RoI = others.define_RoI(configs_input["roi"], chrom_size, configs_input["resolution"])
            if RoI is not None:
                logger.info("region of interest to be used for plotting: %s:%d-%d", chrom_name, *RoI["genomic"])

            logger = logger.bind(step=(1,))
            logger.info("data pre-processing")
            start_time = time.time()
            LT_Iproc, UT_Iproc, Iproc_RoI = stripepy.step_1(
                I,
                configs_input["genomic_belt"],
                configs_input["resolution"],
                RoI=RoI,
                logger=logger,
            )
            progress_bar(progress_weights["step_1"])
            logger.info("preprocessing took %s", pretty_format_elapsed_time(start_time))

            # Find the indices where the sum is zero
            # TODO: DO SOMETHING
            # zero_indices = np.where(np.sum(Iproc_RoI, axis=0) == 0)[0]
            # print(np.min(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # print(np.max(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # np.savetxt("trend.txt", np.sum(LT_Iproc + UT_Iproc, axis=0))
            # exit()

            logger = logger.bind(step=(2,))
            logger.info("topological data analysis")
            start_time = time.time()
            result = stripepy.step_2(
                chrom_name,
                f.chromosomes().get(chrom_name),
                LT_Iproc,
                UT_Iproc,
                configs_thresholds["glob_pers_min"],
                logger=logger,
            )
            progress_bar(progress_weights["step_2"])
            logger.info("topological data analysis took %s", pretty_format_elapsed_time(start_time))

            if RoI is not None:
                result.set_roi(RoI)

            logger = logger.bind(step=(3,))
            logger.info("shape analysis")
            start_time = time.time()
            result = stripepy.step_3(
                result,
                LT_Iproc,
                UT_Iproc,
                configs_input["resolution"],
                configs_input["genomic_belt"],
                configs_thresholds["max_width"],
                configs_thresholds["loc_pers_min"],
                configs_thresholds["loc_trend_min"],
                map=pool.map if pool is not None else map,
                num_chunks=configs_other["nproc"],
                logger=logger,
            )

            progress_bar(progress_weights["step_3"])
            logger.info("shape analysis took %s", pretty_format_elapsed_time(start_time))

            logger = logger.bind(step=(4,))
            logger.info("statistical analysis and post-processing")
            start_time = time.time()

            result = stripepy.step_4(
                result,
                LT_Iproc,
                UT_Iproc,
                num_chunks=configs_other["nproc"],
                map_=pool.map if pool is not None else map,
                logger=logger,
            )

            progress_bar(progress_weights["step_4"])
            logger.info("statistical analysis and post-processing took %s", pretty_format_elapsed_time(start_time))

            logger = main_logger.bind(chrom=chrom_name)
            logger.info('writing results to file "%s"', h5.path)
            h5.write_descriptors(result)
            progress_bar(progress_weights["output"])
            logger.info("processing took %s", pretty_format_elapsed_time(start_local_time))

            if result.roi is not None:
                start_time = time.time()
                logger = logger.bind(step=(5,))
                logger.info("generating plots")
                stripepy.step_5(
                    result,
                    configs_input["resolution"],
                    LT_Iproc,
                    UT_Iproc,
                    f.fetch(
                        f"{chrom_name}:{result.roi['genomic'][0]}-{result.roi['genomic'][1]}",
                        normalization=configs_input["normalization"],
                    ).to_numpy("full"),
                    Iproc_RoI,
                    configs_input["genomic_belt"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    configs_output["output_folder"] / "plots",
                    map=pool.map if pool is not None else map,
                    logger=logger,
                )

                progress_bar(progress_weights["step_5"])
                logger.info("plotting took %s", pretty_format_elapsed_time(start_time))

    main_logger.info("DONE!")
    main_logger.info(
        "processed %d chromosomes in %s", len(f.chromosomes()), pretty_format_elapsed_time(start_global_time)
    )

    return 0
