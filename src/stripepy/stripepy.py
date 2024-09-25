import sys
import time

import h5py
import numpy as np

sys.path.insert(1, "./utils")
import utils.cli as cli
import utils.IO as IO
import utils.others as others
import utils.stripepy as stripepy

MIN_SIZE_CHROMOSOME = 2000000


def save_terminal_groups(name, obj):
    if isinstance(obj, h5py.Group):
        has_subgroups = any(isinstance(child_obj, h5py.Group) for _, child_obj in obj.items())
        if not has_subgroups:
            terminal_group_names.append(name)


def save_all_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):  # check if obj is a group or dataset
        dataset_names.append(name)


def print_all_attributes(obj, parent=""):
    if isinstance(obj, h5py.Group) or isinstance(obj, h5py.File):
        for key, val in obj.attrs.items():
            print(f"{parent}/{key}: {val}")
        for key, sub_obj in obj.items():
            print_all_attributes(sub_obj, f"{parent}/{key}")
    elif isinstance(obj, h5py.Dataset):
        for key, val in obj.attrs.items():
            print(f"{parent}/{key}: {val}")


class Stripepy:
    def __init__():

        # How long does stripepy take to analyze the whole Hi-C matrix?
        start_global_time = time.time()

        # Retrieve input parameters:
        configs_input, configs_thresholds, configs_output = cli.parse_args()

        # Data loading:
        c, chr_starts, chr_ends, bp_lengths = others.cmap_loading(
            configs_input["contact-map"], configs_input["resolution"]
        )

        # Remove existing folders:
        configs_output["output_folder"] = f"{configs_output['output_folder']}/{configs_input['resolution']}"
        IO.remove_and_create_folder(configs_output["output_folder"])

        # Extract a list of tuples where each tuple is (index, chr), e.g. (2,'chr3'):
        c_pairs = others.chromosomes_to_study(list(c.chromosomes().keys()), bp_lengths, MIN_SIZE_CHROMOSOME)

        # Create HDF5 file to store candidate stripes:
        hf = h5py.File(f"{configs_output['output_folder']}/results.hdf5", "w")

        # Keep track of all input parameters:
        hf.attrs["genomic-belt"] = configs_input["genomic_belt"]
        hf.attrs["max-width"] = configs_thresholds["max_width"]
        hf.attrs["constrain-heights"] = configs_thresholds["constrain_heights"]
        hf.attrs["local-persistence-minimum"] = configs_thresholds["loc_pers_min"]
        hf.attrs["local-trend-minimum"] = configs_thresholds["loc_trend_min"]

        # Lopping over all chromosomes:
        for this_chr_idx, this_chr in c_pairs:

            print(f"\n{IO.ANSI.RED}CHROMOSOME {this_chr}{IO.ANSI.ENDC}")
            start_local_time = time.time()

            # Create a group for current chromosome:
            hf.create_group(f"{this_chr}/")

            # Removing and creating folders to store output files:
            # configs_input['roi'] = None
            if configs_input["roi"] is not None:
                IO.create_folders_for_plots(f"{configs_output['output_folder']}/plots/{this_chr}")

            # Extract current chromosome (only the upper-triangular part is stored by hictkpy!):
            I = c.fetch(this_chr).to_coo().tolil()
            I += I.T
            I.setdiag(I.diagonal() / 2)
            I = I.tocsr()

            # RoI:
            RoI = others.define_RoI(
                configs_input["roi"], chr_starts[this_chr_idx], chr_ends[this_chr_idx], configs_input["resolution"]
            )
            print(f"RoI is: {RoI}")

            print(f"{IO.ANSI.YELLOW}Step 1: pre-processing step{IO.ANSI.ENDC}")
            start_time = time.time()
            if all(param is not None for param in [RoI, configs_output["output_folder"]]):
                output_folder_1 = f"{configs_output['output_folder']}/plots/{this_chr}/1_preprocessing/"
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
            # Create the output of this step:
            hf.create_group(f"{this_chr}/global-pseudo-distributions/LT/")
            hf.create_group(f"{this_chr}/global-pseudo-distributions/UT/")
            start_time = time.time()
            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_2 = f"{configs_output['output_folder']}/plots/{this_chr}/2_TDA/"
                pseudo_distributions, candidate_stripes = stripepy.step_2(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_type"],
                    configs_thresholds["glob_pers_min"],
                    hf[f"{this_chr}/global-pseudo-distributions/"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_2,
                )
            else:
                pseudo_distributions, candidate_stripes = stripepy.step_2(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_thresholds["glob_pers_type"],
                    configs_thresholds["glob_pers_min"],
                    hf[f"{this_chr}/global-pseudo-distributions/"],
                )
            print(f"Execution time of step 2: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 3: Shape analysis{IO.ANSI.ENDC}")
            hf.create_group(f"{this_chr}/stripes/LT/")
            hf.create_group(f"{this_chr}/stripes/UT/")
            start_time = time.time()
            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_3 = f"{configs_output['output_folder']}/plots/{this_chr}/3_shape_analysis/"
                stripepy.step_3(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    pseudo_distributions,
                    candidate_stripes,
                    hf[f"{this_chr}/stripes/"],
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_3,
                )
            else:
                stripepy.step_3(
                    LT_Iproc,
                    UT_Iproc,
                    configs_input["resolution"],
                    configs_input["genomic_belt"],
                    configs_thresholds["max_width"],
                    configs_thresholds["constrain_heights"],
                    configs_thresholds["loc_pers_min"],
                    configs_thresholds["loc_trend_min"],
                    pseudo_distributions,
                    candidate_stripes,
                    hf[f"{this_chr}/stripes/"],
                )

            print(f"Execution time of step 3: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.YELLOW}Step 4: Statistical analysis and post-processing{IO.ANSI.ENDC}")
            start_time = time.time()

            if all(param is not None for param in [Iproc_RoI, RoI, configs_output["output_folder"]]):
                output_folder_4 = f"{configs_output['output_folder']}/plots/{this_chr}/4_biological_analysis/"
                thresholds_relative_change = np.arange(0.0, 15.2, 0.2)
                stripepy.step_4(
                    LT_Iproc,
                    UT_Iproc,
                    candidate_stripes,
                    hf[f"{this_chr}/stripes/"],
                    configs_input["resolution"],
                    thresholds_relative_change,
                    Iproc_RoI=Iproc_RoI,
                    RoI=RoI,
                    output_folder=output_folder_4,
                )
            else:
                stripepy.step_4(LT_Iproc, UT_Iproc, candidate_stripes, hf[f"{this_chr}/stripes/"])

            print(f"Execution time of step 4: {time.time() - start_time} seconds ---")

            print(f"{IO.ANSI.CYAN}This chromosome has taken {(time.time() - start_local_time)} seconds{IO.ANSI.ENDC}")

            # # Get all terminal groups within the file
            # # Recover all terminal group names:
            # terminal_group_names = []
            # hf.visititems(save_terminal_groups)
            # print("List of terminal group names:", terminal_group_names)
            #
            # print_all_attributes(hf)
            #
            # exit()

        print(f"\n\n{IO.ANSI.RED}The code has run for {(time.time() - start_global_time) / 60} minutes{IO.ANSI.ENDC}")
        hf.close()
