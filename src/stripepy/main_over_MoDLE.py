import sys
import time

import h5py
import numpy as np

sys.path.append("utils")

import utils.IO as IO
import utils.others as others
import utils.stripepy as stripepy

if __name__ == "__main__":

    # Loop over resolutions:
    for resolution in [5000, 10000, 25000, 50000]:

        # Loop over contact densities:
        for TMP1 in ["1", "5", "10", "15"]:

            # Loop over levels of noise:
            for TMP2 in ["5000", "10000", "15000", "20000"]:

                filename = f"grch38_h1_rad21_{TMP1}_{TMP2}"

                print(f"{IO.ANSI.RED}-------------------------------------------------{IO.ANSI.ENDC}")
                print(f"{IO.ANSI.RED}-------------------------------------------------{IO.ANSI.ENDC}")
                print(f"\n{IO.ANSI.RED}{filename}-{resolution}{IO.ANSI.ENDC}")
                print(f"{IO.ANSI.RED}-------------------------------------------------{IO.ANSI.ENDC}")
                print(f"{IO.ANSI.RED}-------------------------------------------------{IO.ANSI.ENDC}")

                configs_input = {
                    "contact-map": (
                        f"/Users/andreraf/UiO Dropbox/Andrea Raffo/"
                        f"2022-detection-of-architectural-stripes/modle/andreas/"
                        f"MoDLE-benchmark/data/{filename}/"
                        f"{filename}.mcool"
                    ),
                    "resolution": resolution,
                    "genomic_belt": 5000000,
                }

                configs_thresholds = {
                    "glob_pers_type": "constant",
                    "glob_pers_min": 0.025,
                    "constrain_heights": True,
                    "loc_pers_min": 0.25,
                    "loc_trend_min": 0.05,
                    "max_width": 10,
                }

                configs_output = {
                    "output_folder": f"/Users/andreraf/UiO Dropbox/Andrea Raffo/"
                    f"2022-detection-of-architectural-stripes/runs/stripepy/"
                    f"MoDLE-benchmark/{filename}"
                }

                # How long does stripepy take to analyze the whole Hi-C matrix?
                start_global_time = time.time()

                # Data loading:
                c, chr_starts, chr_ends, bp_lengths = others.cmap_loading(
                    configs_input["contact-map"], configs_input["resolution"]
                )

                # Remove existing folders:
                configs_output["output_folder"] = f"{configs_output['output_folder']}/{configs_input['resolution']}"
                IO.remove_and_create_folder(configs_output["output_folder"])

                # Extract a list of tuples where each tuple is (index, chr), e.g. (2,'chr3'):
                MIN_SIZE_CHROMOSOME = 0
                c_pairs = others.chromosomes_to_study(list(c.chromosomes().keys()), bp_lengths, MIN_SIZE_CHROMOSOME)

                # Create HDF5 file to store candidate stripes:
                hf = h5py.File(f"{configs_output['output_folder']}/results.hdf5", "w")

                # Keep track of all input parameters:
                hf.attrs["genomic-belt"] = configs_input["genomic_belt"]
                hf.attrs["max-width"] = configs_thresholds["max_width"]
                hf.attrs["constrain-heights"] = configs_thresholds["constrain_heights"]
                hf.attrs["local-persistence-minimum"] = configs_thresholds["loc_pers_min"]
                hf.attrs["local-trend-minimum"] = configs_thresholds["loc_trend_min"]

                # Print the used parameters (chosen or default-ones):
                print("\nArguments:")
                print(f"--contact-map: {configs_input['contact-map']}")
                print(f"--resolution: {configs_input['resolution']}")
                print(f"--genomic-belt: {configs_input['genomic_belt']}")
                print(f"--max-width: {configs_thresholds['max_width']}")
                print(f"--glob-pers-type: {configs_thresholds['glob_pers_type']}")
                print(f"--glob-pers-min: {configs_thresholds['glob_pers_min']}")
                print(f"--constrain-heights: {configs_thresholds['constrain_heights']}")
                print(f"--loc-pers-min: {configs_thresholds['loc_pers_min']}")
                print(f"--loc-trend-min: {configs_thresholds['loc_trend_min']}")
                print(f"--output-folder: {configs_output['output_folder']}")

                # Lopping over all chromosomes:
                for this_chr_idx, this_chr in c_pairs:

                    print(f"\n\n{IO.ANSI.RED}CHROMOSOME {this_chr}{IO.ANSI.ENDC}")
                    start_local_time = time.time()

                    # Create a group for current chromosome:
                    hf.create_group(f"{this_chr}/")

                    # Region of Interest (RoI) in genomic and matrix coordinates:
                    RoI = dict()
                    e1 = 0
                    e2 = e1 + 2000000
                    RoI["genomic"] = [e1, e2, e1, e2]
                    RoI["matrix"] = [int(roi / configs_input["resolution"]) for roi in RoI["genomic"]]
                    configs_input["roi"] = RoI

                    # Removing and creating folders to store output files:
                    # configs_input['roi'] = None
                    if configs_input["roi"] is not None:
                        IO.create_folders_for_plots(f"{configs_output['output_folder']}/plots/{this_chr}")

                    # Extract current chromosome (only the upper-triangular part is stored by hictkpy!):
                    I = c.fetch(this_chr).to_coo().tolil()
                    I += I.T
                    # I.setdiag(I.diagonal()/2)
                    I = I.tocsr()

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
                        LT_Iproc, UT_Iproc, _ = stripepy.step_1(
                            I, configs_input["genomic_belt"], configs_input["resolution"]
                        )
                        Iproc_RoI = None
                    print(f"Execution time of step 1: {time.time() - start_time} seconds ---")

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

                    print(
                        f"{IO.ANSI.CYAN}This chromosome has taken {(time.time() - start_local_time)} seconds{IO.ANSI.ENDC}"
                    )

                print(
                    f"\n\n{IO.ANSI.RED}The code has run for {(time.time() - start_global_time) / 60} minutes{IO.ANSI.ENDC}"
                )
                hf.close()
