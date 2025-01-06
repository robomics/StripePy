# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import json
import multiprocessing as mp
import pathlib
import sys
import time
from typing import Any, Dict

import alive_progress as ap
import numpy as np
import pandas as pd
import structlog

from stripepy import IO, others, stripepy


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


def _compute_progress_bar_weights(chrom_sizes: Dict[str, int]) -> pd.DataFrame:
    # These weights have been computed on a Linux machine (Ryzen 9 5950X) using 1 core to process
    # 4DNFI9GMP2J8.mcool at 10kbp
    step_weights = {
        "input": 0.067616,
        "step_1": 0.039959,
        "step_2": 0.033109,
        "step_3": 0.721486,
        "step_4": 0.135708,
        "output": 0.002122,
    }

    assert np.isclose(sum(step_weights.values()), 1.0)

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


def run(
    configs_input: Dict[str, Any],
    configs_thresholds: Dict[str, Any],
    configs_output: Dict[str, Any],
    configs_other: Dict[str, Any],
):
    # How long does stripepy take to analyze the whole Hi-C matrix?
    start_global_time = time.time()

    _write_param_summary(configs_input, configs_thresholds, configs_output, configs_other)

    # Data loading:
    f, chr_starts, chr_ends, bp_lengths = others.cmap_loading(configs_input["contact_map"], configs_input["resolution"])

    # Remove existing folders:
    # configs_output["output_folder"] = (
    #     f"{configs_output['output_folder']}/{configs_input['contact_map'].stem}/{configs_input['resolution']}"
    # )
    IO.remove_and_create_folder(configs_output["output_folder"], configs_output["force"])

    # Extract a list of tuples where each tuple is (index, chr), e.g. (2,'chr3'):
    c_pairs = others.chromosomes_to_study(
        list(f.chromosomes().keys()), bp_lengths, configs_thresholds["min_chrom_size"]
    )

    with contextlib.ExitStack() as ctx:
        main_logger = structlog.get_logger()

        # Create HDF5 file to store candidate stripes:
        main_logger.info('initializing result file "%s"...', configs_output["output_folder"] / "results.hdf5")
        h5 = ctx.enter_context(IO.ResultFile(configs_output["output_folder"] / "results.hdf5", "w"))

        h5.init_file(f, configs_input["normalization"], _generate_metadata_attribute(configs_input, configs_thresholds))

        # Set up the process pool when appropriate
        if configs_other["nproc"] > 1:
            main_logger.debug("initializing a pool of %d processes...", configs_other["nproc"])
            pool = ctx.enter_context(mp.Pool(configs_other["nproc"]))
        else:
            pool = None

        disable_bar = not sys.stderr.isatty()
        progress_weights = _compute_progress_bar_weights(f.chromosomes())
        progress_bar = ctx.enter_context(
            ap.alive_bar(
                total=sum([(chr_ends[i] - chr_starts[i]) * f.resolution() for i, _ in c_pairs]),
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
        for this_chr_idx, this_chr in c_pairs:
            logger = main_logger.bind(chrom=this_chr)
            logger.info("begin processing...")
            start_local_time = time.time()

            pw0, pw1, pw2, pw3, pw4, pw5 = progress_weights.loc[this_chr, :]

            # Removing and creating folders to store output files:
            # configs_input['roi'] = None
            if configs_input["roi"] is not None:
                IO.create_folders_for_plots(configs_output["output_folder"] / "plots" / this_chr)

            logger.debug("fetching interactions using normalization=%s", configs_input["normalization"])
            I = f.fetch(this_chr, normalization=configs_input["normalization"]).to_csr("full")
            progress_bar(pw0)

            # RoI:
            RoI = others.define_RoI(
                configs_input["roi"], chr_starts[this_chr_idx], chr_ends[this_chr_idx], configs_input["resolution"]
            )
            if RoI is not None:
                logger.info("region of interest to be used for plotting: %s", RoI)

            logger = logger.bind(step=(1,))
            logger.info("data pre-processing")
            start_time = time.time()
            if all(param is not None for param in [RoI, configs_output["output_folder"]]):
                output_folder_1 = f"{configs_output['output_folder']}/plots/{this_chr}/1_preprocessing/"
                LT_Iproc, UT_Iproc, Iproc_RoI = stripepy.step_1(
                    I,
                    configs_input["genomic_belt"],
                    configs_input["resolution"],
                    RoI=RoI,
                    output_folder=output_folder_1,
                    logger=logger,
                )
            else:
                LT_Iproc, UT_Iproc, _ = stripepy.step_1(
                    I, configs_input["genomic_belt"], configs_input["resolution"], logger=logger
                )
                Iproc_RoI = None
            progress_bar(pw1)
            logger.info("preprocessing took %s seconds", time.time() - start_time)

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
            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_2 = f"{configs_output['output_folder']}/plots/{this_chr}/2_TDA/"
                result = stripepy.step_2(
                    this_chr,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_2,
                    logger=logger,
                )
            else:
                result = stripepy.step_2(
                    this_chr,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                    logger=logger,
                )
            progress_bar(pw2)
            logger.info("topological data analysis took %s seconds", time.time() - start_time)

            logger = logger.bind(step=(3,))
            logger.info("shape analysis")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_3 = f"{configs_output['output_folder']}/plots/{this_chr}/3_shape_analysis/"
                result = stripepy.step_3(
                    result,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_3,
                    map=pool.map if pool is not None else map,
                    logger=logger,
                )
            else:
                result = stripepy.step_3(
                    result,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    map=pool.map if pool is not None else map,
                    logger=logger,
                )

            progress_bar(pw3)
            logger.info("shape analysis took %s seconds", time.time() - start_time)

            logger = logger.bind(step=(4,))
            logger.info("statistical analysis and post-processing")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_4 = f"{configs_output['output_folder']}/plots/{this_chr}/4_biological_analysis/"
                thresholds_relative_change = np.arange(0.0, 15.2, 0.2)
                result = stripepy.step_4(
                    result,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    thresholds_relative_change,
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_4,
                    logger=logger,
                )
            else:
                result = stripepy.step_4(
                    result,
                    LT_Iproc,
                    UT_Iproc,
                    logger=logger,
                )

            progress_bar(pw4)
            logger.info("statistical analysis and post-processing took %s seconds", time.time() - start_time)

            logger = main_logger.bind(chrom=this_chr)
            logger.info('writing results for "%s" to file "%s"', this_chr, h5.path)
            h5.write_descriptors(result)
            progress_bar(pw5)
            logger.info("processing took %s seconds", time.time() - start_local_time)

    main_logger.info("DONE!")
    main_logger.info(
        "processed %d chromosomes in %s minutes", len(f.chromosomes()), (time.time() - start_global_time) / 60
    )
