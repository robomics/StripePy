# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import multiprocessing as mp
import time
from typing import Any, Dict, List, Tuple

import numpy as np

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


def _plan(chromosomes: Dict[str, int], min_size: int) -> List[Tuple[str, int, bool]]:
    plan = []
    small_chromosomes = []
    for chrom, length in chromosomes.items():
        skip = length <= min_size
        plan.append((chrom, length, skip))
        if skip:
            small_chromosomes.append(chrom)

    if len(small_chromosomes) != 0:
        print(
            f"{IO.ANSI.RED}ATT: The following chromosomes are discarded because shorter than --min-chrom-size = "
            f"{min_size} bp: {', '.join(small_chromosomes)}{IO.ANSI.ENDC}"
        )

    return plan


def generate_empty_result(chrom: str, chrom_size: int, resolution: int) -> IO.Result:
    result = IO.Result(chrom)
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


def run(
    configs_input: Dict[str, Any],
    configs_thresholds: Dict[str, Any],
    configs_output: Dict[str, Any],
    configs_other: Dict[str, Any],
):
    # How long does stripepy take to analyze the whole Hi-C matrix?
    start_global_time = time.time()

    # Data loading:
    f = others.open_matrix_file_checked(configs_input["contact_map"], configs_input["resolution"])

    # Remove existing folders:
    # configs_output["output_folder"] = (
    #     f"{configs_output['output_folder']}/{configs_input['contact_map'].stem}/{configs_input['resolution']}"
    # )
    configs_output["output_folder"] = (
        configs_output["output_folder"] / configs_input["contact_map"].stem / str(configs_input["resolution"])
    )
    IO.remove_and_create_folder(configs_output["output_folder"], configs_output["force"])

    with contextlib.ExitStack() as ctx:
        # Create HDF5 file to store candidate stripes:
        h5 = ctx.enter_context(IO.ResultFile(configs_output["output_folder"] / "results.hdf5", "w"))

        h5.init_file(f, configs_input["normalization"], _generate_metadata_attribute(configs_input, configs_thresholds))

        # Set up the process pool when appropriate
        if configs_other["nproc"] > 1:
            pool = ctx.enter_context(mp.Pool(configs_other["nproc"]))
        else:
            pool = None

        # Lopping over all chromosomes:
        for chrom_name, chrom_size, skip in _plan(
            f.chromosomes(include_ALL=False), configs_thresholds["min_chrom_size"]
        ):
            if skip:
                result = generate_empty_result(chrom_name, chrom_size, configs_input["resolution"])
                h5.write_descriptors(result)
                continue

            print(f"\n{IO.ANSI.RED}CHROMOSOME {chrom_name}{IO.ANSI.ENDC}")
            start_local_time = time.time()

            # Removing and creating folders to store output files:
            # configs_input['roi'] = None
            if configs_input["roi"] is not None:
                IO.create_folders_for_plots(configs_output["output_folder"] / "plots" / chrom_name)

            I = f.fetch(chrom_name, normalization=configs_input["normalization"]).to_csr("full")

            # RoI:
            RoI = others.define_RoI(configs_input["roi"], chrom_size, configs_input["resolution"])
            print(f"RoI is: {RoI}")

            print(f"{IO.ANSI.YELLOW}Step 1: pre-processing step{IO.ANSI.ENDC}")
            start_time = time.time()
            if all(param is not None for param in [RoI, configs_output["output_folder"]]):
                output_folder_1 = f"{configs_output['output_folder']}/plots/{chrom_name}/1_preprocessing/"
                LT_Iproc, UT_Iproc, Iproc_RoI = stripepy.step_1(
                    I,
                    configs_input["genomic_belt"],
                    configs_input["resolution"],
                    RoI=RoI,
                    output_folder=output_folder_1,
                )
            else:
                LT_Iproc, UT_Iproc, _ = stripepy.step_1(I, configs_input["genomic_belt"], configs_input["resolution"])
                Iproc_RoI = None
            print(f"Execution time of step 1: {time.time() - start_time} seconds ---")

            # Find the indices where the sum is zero
            # TODO: DO SOMETHING
            # zero_indices = np.where(np.sum(Iproc_RoI, axis=0) == 0)[0]
            # print(np.min(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # print(np.max(np.sum(LT_Iproc + UT_Iproc, axis=0)))
            # np.savetxt("trend.txt", np.sum(LT_Iproc + UT_Iproc, axis=0))
            # exit()

            print(f"{IO.ANSI.YELLOW}Step 2: Topological Data Analysis{IO.ANSI.ENDC}")
            start_time = time.time()
            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_2 = f"{configs_output['output_folder']}/plots/{chrom_name}/2_TDA/"
                result = stripepy.step_2(
                    chrom_name,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_2,
                )
            else:
                result = stripepy.step_2(
                    chrom_name,
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_min"],
                )
            print(f"Execution time of step 2: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 3: Shape analysis{IO.ANSI.ENDC}")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_3 = f"{configs_output['output_folder']}/plots/{chrom_name}/3_shape_analysis/"
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
                )

            print(f"Execution time of step 3: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 4: Statistical analysis and post-processing{IO.ANSI.ENDC}")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_4 = f"{configs_output['output_folder']}/plots/{chrom_name}/4_biological_analysis/"
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
                )
            else:
                result = stripepy.step_4(result, LT_Iproc, UT_Iproc)

            print(f"Execution time of step 4: {time.time() - start_time} seconds ---")

            print(f'Writing results for "{chrom_name}" to file "{h5.path}"...')
            h5.write_descriptors(result)

            print(f"{IO.ANSI.CYAN}This chromosome has taken {(time.time() - start_local_time)} seconds{IO.ANSI.ENDC}")

    print(f"\n\n{IO.ANSI.RED}The code has run for {(time.time() - start_global_time) / 60} minutes{IO.ANSI.ENDC}")
