import itertools
import sys

import bioframe as bf
import hictkpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append("utils")
import IO
from utils.evaluate import *

# Colors
colors = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f", "#669bbc"]


def initialize():
    """
    Initialize dictionaries for the five runs (four methods, stripepy has two default configurations). We use the
    following encoding: M1=stripepy-nf, M2=stripepy-f, M3=chromosight, M4=stripecaller, M5=stripenn.
    :return: the dictionaries M1, M2, M3, M4, M5
    """
    keys1 = ["is_candidate_good", "is_anchor_found", "classification_vector", "GT_classification_vector"]
    keys2 = ["TN", "FP", "FN", "TP"]
    M1, M2, M3, M4, M5 = [{**{key: [] for key in keys1}, **{key: 0 for key in keys2}} for _ in range(5)]

    return M1, M2, M3, M4, M5


def retrieve_ground_truth(path, file_name, chromosome, resolution, probability_threshold, n_bins):
    """
    This function defines a dictionary that contains ground truth (GT) information.
    :param path: path to the ground truth file
    :param file_name: name of the bed file containing coordinates and occupancy of each extrusion barrier
    :param chromosome: chromosome name
    :param resolution: resolution of the contact map
    :param probability_threshold: cut-off value for the occupancy
    :param n_bins: number of bins for current chromosome at given resolution -- used to define a classification vector
    :return: dictionary GT containing location of anchor points and classification vectors.
    """

    # Ground truth for the chromosome under study:
    df = bf.read_table(path + file_name, schema="bed6")
    df_sub = bf.select(df, chromosome).copy()
    df_sub["anchor"] = (df_sub["end"] + df_sub["start"]) / 2

    # Lower ground truth stripes:
    LT_stripes = df_sub[(df_sub.score > probability_threshold) & (df_sub.strand == "+")]

    # Upper ground truth stripes:
    UT_stripes = df_sub[(df_sub.score > probability_threshold) & (df_sub.strand == "-")]

    # Retrieve ground truth anchors
    GT = dict()
    GT["L_anchors"] = np.unique(np.round(LT_stripes.anchor.values / resolution)).astype(int).tolist()
    GT["U_anchors"] = np.unique(np.round(UT_stripes.anchor.values / resolution)).astype(int).tolist()

    # Classification vectors:
    GT["number_of_bins"] = n_bins
    GT["L_clas_vec"] = np.isin(np.arange(n_bins), GT["L_anchors"]).astype(int).tolist()
    GT["U_clas_vec"] = np.isin(np.arange(n_bins), GT["U_anchors"]).astype(int).tolist()
    GT["clas_vec"] = GT["L_clas_vec"] + GT["U_clas_vec"]

    return GT


def retrieve_stripepy(path, chromosome, n_bins, resolution, threshold):
    """
    Retrieve prediction from stripepy (stripepy-nf: threshold set to 0, stripepy-f: threshold set to 10.0)
    :param path: path to the folder containing the predictions for the current Hi-C map
    :param chromosome: chromosome name
    :param n_bins: number of bins for current chromosome at given resolution -- used to define a classification vector
    :param resolution: resolution of the contact map
    :param threshold: cut-off value for the relative change parameter
    :return: three tuples:
                (*) (L_HIoIs, L_VIoIs) contains the lower-triangular stripes;
                (*) (U_HIoIs, U_VIoIs) contains the upper-triangular stripes;
                (*) (L_clas_vec, U_clas_vec) contains the clas. vectors for the lower- and upper-triangular parts.
    """

    # Update path to point at current resolution and chromosome:
    path = f"{path}/{resolution}/{chromosome}"

    # Candidate horizontal/vertical intervals of interest (each interval defines a candidate stripe):
    L_IoIs = bf.read_table(f"{path}/global/filtrations/LT_{threshold:.2f}.bedpe", schema="bed6")
    U_IoIs = bf.read_table(f"{path}/global/filtrations/UT_{threshold:.2f}.bedpe", schema="bed6")

    # Convert IoIs to NumPy arrays and divide by resolution:
    L_HIoIs = (L_IoIs[["start", "end"]].values / resolution).astype(int)
    L_VIoIs = (L_IoIs[["score", "strand"]].values / resolution).astype(int)
    U_HIoIs = (U_IoIs[["start", "end"]].values / resolution).astype(int)
    U_VIoIs = (U_IoIs[["score", "strand"]].values / resolution).astype(int)

    # Seeds (we load all of them and then keep just those in a surviving candidate stripe)...
    # the output should be re-arranged to avoid this!
    Ldf = pd.read_csv(f"{path}/global/all/LT_shape-descriptors.csv", dtype=int)
    Udf = pd.read_csv(f"{path}/global/all/UT_shape-descriptors.csv", dtype=int)

    L_seeds = Ldf[Ldf["seed"].isin([i for L_HIoI in L_HIoIs for i in range(L_HIoI[0], L_HIoI[1] + 1)])]["seed"].tolist()
    U_seeds = Udf[Udf["seed"].isin([i for U_HIoI in U_HIoIs for i in range(U_HIoI[0], U_HIoI[1] + 1)])]["seed"].tolist()

    # Classification vectors (predicted):
    L_clas_vec = np.where(np.isin(range(n_bins), L_seeds), 1, 0)
    U_clas_vec = np.where(np.isin(range(n_bins), U_seeds), 1, 0)

    return (L_HIoIs, L_VIoIs), (U_HIoIs, U_VIoIs), (L_clas_vec, U_clas_vec)


def retrieve_chromosight(path, chromosome, n_bins, resolution):
    """
    Retrieve prediction from Chromosight
    :param path: path to the folder containing the predictions for the current Hi-C map
    :param chromosome: chromosome name
    :param n_bins: number of bins for current chromosome at given resolution -- used to define a classification vector
    :param resolution: resolution of the contact map
    :return: three tuples:
                (*) (L_HIoIs, L_VIoIs) contains the lower-triangular stripes;
                (*) (U_HIoIs, U_VIoIs) contains the upper-triangular stripes;
                (*) (L_clas_vec, U_clas_vec) contains the clas. vectors for the lower- and upper-triangular parts.
    """

    # Load predictions -- lower-triangular:
    Ldf = pd.read_csv(f"{path}/{resolution}/left/output.tsv", sep="\t")
    Ldf_chr = Ldf[Ldf["chrom1"] == chromosome]

    # Load predictions -- upper-triangular:
    Udf = pd.read_csv(f"{path}/{resolution}/right/output.tsv", sep="\t")
    Udf_chr = Udf[Udf["chrom1"] == chromosome]

    # Gather lower- and upper-triangular candidates:
    # NB: Chromosight does not estimate width and length, but just gives a genomic pair (point, pixel) of where a stripe
    # might lie. For visualization purposes, end values are the start values increased by the resolution value.
    # For evaluation purposes, we just select the start values.
    L_HIoIs = (Ldf_chr[["start1", "start1"]].values / resolution).astype(int).tolist()
    L_VIoIs = (Ldf_chr[["start2", "start2"]].values / resolution).astype(int).tolist()
    U_HIoIs = (Udf_chr[["start2", "start2"]].values / resolution).astype(int).tolist()
    U_VIoIs = (Udf_chr[["start1", "start1"]].values / resolution).astype(int).tolist()

    # Gather anchors:
    L_anchors = [x[0] for x in L_HIoIs]
    U_anchors = [x[0] for x in U_HIoIs]

    # Classification vectors (predicted):
    L_clas_vec = np.where(np.isin(range(n_bins), L_anchors), 1, 0)
    U_clas_vec = np.where(np.isin(range(n_bins), U_anchors), 1, 0)

    return (L_HIoIs, L_VIoIs), (U_HIoIs, U_VIoIs), (L_clas_vec, U_clas_vec)


def retrieve_stripecaller(path, chromosome, n_bins, resolution):
    """
    Retrieve prediction from StripeCaller
    :param path: path to the folder containing the predictions for the current Hi-C map
    :param chromosome: chromosome name
    :param n_bins: number of bins for current chromosome at given resolution -- used to define a classification vector
    :param resolution: resolution of the contact map
    :return: three tuples:
                (*) (L_HIoIs, L_VIoIs) contains the lower-triangular stripes;
                (*) (U_HIoIs, U_VIoIs) contains the upper-triangular stripes;
                (*) (L_clas_vec, U_clas_vec) contains the clas. vectors for the lower- and upper-triangular parts.
    """

    # Load predictions:
    df = bf.read_table(f"{path}/{resolution}/output.bedpe", schema="bed6")
    df_chr = df[df["chrom"] == chromosome]
    X1 = df_chr["start"].values.tolist()
    X2 = df_chr["end"].values.tolist()
    Y1 = df_chr["score"].values.tolist()
    Y2 = df_chr["strand"].values.tolist()

    # NB: StripeCaller does not estimate the width of a stripe, but constrained it to 1 bin. For visualization purposes,
    # end values are the start values increased by the resolution value. For evaluation purposes, we just select the
    # start values.

    # Gather lower-triangular candidates:
    L_ids = [i for i, (x1, x2, y1, y2) in enumerate(zip(X1, X2, Y1, Y2)) if x2 - x1 < y2 - y1]
    L_HIoIs = [[s / resolution, s / resolution] for idx, s in enumerate(X1) if idx in L_ids]
    L_VIoIs = [[s / resolution, e / resolution] for idx, (s, e) in enumerate(zip(Y1, Y2)) if idx in L_ids]

    # Gather upper-triangular candidates:
    U_ids = [i for i, (x1, x2, y1, y2) in enumerate(zip(X1, X2, Y1, Y2)) if y2 - y1 < x2 - x1]
    U_HIoIs = [[s / resolution, s / resolution] for idx, s in enumerate(Y1) if idx in U_ids]
    U_VIoIs = [[s / resolution, e / resolution] for idx, (s, e) in enumerate(zip(X1, X2)) if idx in U_ids]

    # Gather anchors:
    L_anchors = [x[0] for x in L_HIoIs]
    U_anchors = [x[0] for x in U_HIoIs]

    # Classification vectors (predicted):
    L_clas_vec = np.array([1 if i in L_anchors else 0 for i in range(n_bins)])
    U_clas_vec = np.array([1 if i in U_anchors else 0 for i in range(n_bins)])

    return (L_HIoIs, L_VIoIs), (U_HIoIs, U_VIoIs), (L_clas_vec, U_clas_vec)


def retrieve_stripenn(path, chromosome, n_bins, resolution, filter=False):
    """
    Retrieve prediction from Stripenn
    :param path: path to the folder containing the predictions for the current Hi-C map
    :param chromosome: chromosome name
    :param n_bins: number of bins for current chromosome at given resolution -- used to define a classification vector
    :param resolution: resolution of the contact map
    :param filter: if True, it considered the filtered candidates (default is False)
    :return: three tuples:
                (*) (L_HIoIs, L_VIoIs) contains the lower-triangular stripes;
                (*) (U_HIoIs, U_VIoIs) contains the upper-triangular stripes;
                (*) (L_clas_vec, U_clas_vec) contains the clas. vectors for the lower- and upper-triangular parts.
    """

    # Load predictions:
    if filter:
        df = pd.read_csv(f"{path}/{resolution}/result_filtered.tsv", sep="\t")
    else:
        df = pd.read_csv(f"{path}/{resolution}/result_unfiltered.tsv", sep="\t")

    # Filter rows for the specified chromosome and candidate types (lower- and upper-triangular candidates):
    L_df_chr = df[(df["chr"] == chromosome) & (df["pos1"] == df["pos3"])]
    U_df_chr = df[(df["chr"] == chromosome) & (df["pos2"] == df["pos4"])]

    # Gather lower-triangular candidates:
    L_HIoIs = L_df_chr[["pos1", "pos2"]].values / resolution
    L_VIoIs = L_df_chr[["pos3", "pos4"]].values / resolution

    # Gather upper-triangular candidates:
    U_HIoIs = U_df_chr[["pos1", "pos2"]].values / resolution
    U_VIoIs = U_df_chr[["pos3", "pos4"]].values / resolution

    # Gather anchors:
    L_anchors = ((L_df_chr["pos1"] + L_df_chr["pos2"]) / (2 * resolution)).round().astype(int)
    U_anchors = ((U_df_chr["pos1"] + U_df_chr["pos2"]) / (2 * resolution)).round().astype(int)

    # Classification vectors (predicted):
    L_clas_vec = np.where(np.isin(range(n_bins), L_anchors), 1, 0)
    U_clas_vec = np.where(np.isin(range(n_bins), U_anchors), 1, 0)

    return (L_HIoIs, L_VIoIs), (U_HIoIs, U_VIoIs), (L_clas_vec, U_clas_vec)


def is_anchor_in_stripes(GT_anchors, pred_HIoIs):
    """
    Checks: (1) which anchors are contained in a predicted horizontal interval and (2) which predicted horizontal
    intervals contain an anchor
    :param GT_anchors: list of anchors from the ground truth
    :param pred_HIoIs: list of pairs, each pair defines the horizontal "domain" of a stripe
    :return: two lists, is_anchor_found and is_candida_good
    """

    # Initialize arrays for is_anchor_found and is_candida_good with zeros
    is_anchor_found = np.zeros(len(GT_anchors), dtype=int)
    is_candida_good = np.zeros(len(pred_HIoIs), dtype=int)

    # Convert GT_anchors and pred_HIoIs to NumPy arrays for efficient comparison
    GT_anchors_arr = np.array(GT_anchors)
    pred_HIoIs_arr = np.array(pred_HIoIs)

    # Find if each GT anchor is contained in a candidate stripe
    for i, GT_anchor in enumerate(GT_anchors_arr):
        is_in_interval = (
            (GT_anchor >= pred_HIoIs_arr[:, 0]) & (GT_anchor <= pred_HIoIs_arr[:, 1])
            if pred_HIoIs_arr.shape[0] > 0
            else np.array([], dtype=bool)
        )

        if np.any(is_in_interval):
            is_anchor_found[i] = 1
            is_candida_good[is_in_interval] = 1

    return is_anchor_found, is_candida_good


def unique_HIoIs(HIoIs):
    """
    Remove repeating horizontal intervals (produced by Chromosight and StripeCaller):
    :param HIoIs: horizontal intervals of interest
    :return: unique horizontal intervals of interest uHIoIs
    """

    uHIoIs = []
    seen = set()
    for HIoI in HIoIs:
        t_HIoI = tuple(HIoI)
        if t_HIoI not in seen:
            uHIoIs.append(HIoI)
            seen.add(t_HIoI)
    return uHIoIs


def compare_predictions_to_GT(GT, L_HIoIs, U_HIoIs, L_clas_vec, U_clas_vec):
    """
    Compare the horizontal interval of interests and the classification vectors with the ground truth values.
    :param GT: dictionary containing the ground truth values produced by retrieve_ground_truth
    :param L_HIoIs: lower-triangular horizontal intervals of interest
    :param U_HIoIs: upper-triangular horizontal intervals of interest
    :param L_clas_vec: lower-triangular classification vector
    :param U_clas_vec: upper-triangular classification vector
    :return:
    """

    # Remove repeating pairs (Chromosight and StripeCaller break stripes into sub-stripes):
    L_HIoIs = unique_HIoIs(L_HIoIs)
    U_HIoIs = unique_HIoIs(U_HIoIs)

    # Initialize dictionary that will contain information for the lower- upper-triangular matrices:
    M = dict()

    # Check which GT anchors fall into the candidate stripes AND which candidate stripes contain an anchor site:
    L_is_anchor_found, L_is_candidate_good = is_anchor_in_stripes(GT_anchors=GT["L_anchors"], pred_HIoIs=L_HIoIs)
    U_is_anchor_found, U_is_candidate_good = is_anchor_in_stripes(GT_anchors=GT["U_anchors"], pred_HIoIs=U_HIoIs)
    M["is_anchor_found"] = np.concatenate((L_is_anchor_found, U_is_anchor_found)).tolist()
    M["is_candidate_good"] = np.concatenate((L_is_candidate_good, U_is_candidate_good)).tolist()

    # Confusion matrix:
    y_true = GT["L_clas_vec"] + GT["U_clas_vec"]
    y_pred = np.concatenate((L_clas_vec, U_clas_vec))
    M["TN"], M["FP"], M["FN"], M["TP"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return M


def compute_measures(M):
    """
    Compute classification and recognition measures from values stored in the dictionaries produced by
    compare_predictions_to_GT (one dictionary per method).
    :param M: dictionary produced by compare_predictions_to_GT
    :return: classification (TPR, TNR, PPV, bACC, GM, MCC) and recognition (AHR, FGC) measures
    """

    # 1) STRIPE-BASED RECOGNITION MEASURES

    # Anchor Hit Rate:
    AHR = np.sum(M["is_anchor_found"]) / len(M["is_anchor_found"])

    # Fraction Good Candidates:
    FGC = np.sum(M["is_candidate_good"]) / len(M["is_candidate_good"]) if len(M["is_candidate_good"]) > 0 else 0

    # 2) ANCHOR-BASED CLASSIFICATION MEASURES

    # Various combinations for later use:
    X1 = np.float64(M["TP"] + M["FP"])
    X2 = np.float64(M["TP"] + M["FN"])
    X3 = np.float64(M["TN"] + M["FP"])
    X4 = np.float64(M["TN"] + M["FN"])

    # Sensitivity, recall, hit rate, or true positive rate:
    TPR = M["TP"] / (M["TP"] + M["FN"]) if M["TP"] + M["FN"] > 0 else 0

    # Specificity, selectivity, or true negative rate:
    TNR = M["TN"] / (M["TN"] + M["FP"]) if M["TN"] + M["FP"] > 0 else 0

    # Precision or positive predictive value:
    PPV = M["TP"] / (M["TP"] + M["FP"]) if M["TP"] + M["FP"] > 0 else 0

    # Balanced accuracy:
    bACC = 0.5 * (TPR + TNR)

    # G-mean (i.e., geometric mean):
    GM = np.sqrt(TPR * TNR)

    # Matthew's Correlation Coefficient:
    num4MCC = np.float64(M["TP"] * M["TN"] - M["FP"] * M["FN"])
    den4MCC = np.sqrt(X1 * X2 * X3 * X4)
    MCC = num4MCC / den4MCC if den4MCC > 0 else 0

    return TPR, TNR, PPV, bACC, GM, MCC, AHR, FGC


def LaTex_tables(results, resolutions, contact_densities, noises):
    """
    Print the classification and recognition measures for the whole benchmark in the form of a LaTex-friendly table.
    :param results: pandas dataframe containing all measures
    :param resolutions: resolutions in the benchmark
    :param contact_densities: contact densities in the benchmark
    :param noises: noise levels in the benchmark
    :return: -
    """

    # Loop over resolutions:
    for resolution in resolutions:

        print(f"{IO.ANSI.GREEN}Resolution {resolution}{IO.ANSI.ENDC}")

        # Loop over levels of noise:
        for noise in noises:

            # Slice the dataframe:
            sliced_results = results[(results["Resolution"] == resolution) & (results["Noise"] == noise)]

            # Print:
            for num_meas, m in enumerate(["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):
                if num_meas == 0:
                    this_row = (
                        "\\multirow{8}{*}{" f"{round(noise / 1000)}k" "}  & " "\\multicolumn{1}{c|}{" "" f"{m}" "}" " "
                    )
                else:
                    this_row = f" & " "\\multicolumn{1}{c|}{" "" f"{m}" "}" " "
                for contact in contact_densities:
                    for method in ["stripepy1", "stripepy2", "chromosight", "stripecaller", "stripenn"]:
                        this_row += f" & {sliced_results[
                                              (sliced_results["Method"] == method) &
                                              (sliced_results["Contact Density"] == contact)][m].values[0] * 100:>6.2f}"
                this_row += "\\\\"
                print(this_row)
            print("\\hline")


def marginal_plots(results, resolutions, contact_densities, noises, path2output):
    """
    Produce plots that studies the change in each of the three factors: resolution, contact density, and noise.
    :param results: pandas dataframe containing all measures
    :param resolutions: resolutions in the benchmark
    :param contact_densities: contact densities in the benchmark
    :param noises: noise levels in the benchmark
    :param path2output: path to folder where to save output jpg images
    :return: -
    """
    # 1) CHANGE IN RESOLUTION

    # Storing the medians for plots!
    Q2s_by_res = dict()
    for meas in ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]:
        Q2s_by_res[meas] = []

    # Looping in resolution:
    fig, axes = plt.subplots(8, 4, figsize=(6, 9))
    for n_res, resolution in enumerate(resolutions):

        # Extract rows referring to current resolution:
        sliced_results = results.loc[results["Resolution"] == resolution]

        # Medians for current resolution:
        Q2s_this_res = dict()

        for n_meas, clas_meas_name in enumerate(["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):

            # Medians for current resolution and current measure:
            Q2s_this_res[clas_meas_name] = []
            for method in ["stripepy1", "stripepy2", "chromosight", "stripecaller", "stripenn"]:
                Q2s_this_res[clas_meas_name].append(
                    np.median(sliced_results.loc[sliced_results["Method"] == method][clas_meas_name].values)
                )
            Q2s_by_res[clas_meas_name].append(Q2s_this_res[clas_meas_name])

            # Box Plots
            ax = axes[n_meas, n_res]
            sns.boxplot(
                y=clas_meas_name,
                x="Method",
                data=sliced_results,
                ax=ax,
                palette=colors,
                hue="Method",
                orient="v",
                width=0.5,
                fliersize=3,
            )
            ax.set(xlabel=None, ylabel=None)
            ax.tick_params(labelbottom=False, bottom=False)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.axes.get_xaxis().set_visible(False)
            if n_res == 0:
                ax.set_ylabel(clas_meas_name, fontsize=12)
            if n_meas == 0:
                ax.set_title(f"r = {int(resolution / 1000)}k", fontsize=12)
                ax.set_ylim((-0.05, 0.45))
                minor_ticks = np.linspace(0.00, 0.40, 11)
                major_ticks = np.linspace(0.00, 0.40, 6)
            elif n_meas == 1:
                ax.set_ylim((0.90, 1.01))
                minor_ticks = np.linspace(0.90, 1.00, 11)
                major_ticks = np.linspace(0.90, 1.00, 6)
            elif n_meas == 2:
                ax.set_ylim((-0.02, 1.02))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            elif n_meas == 3:
                ax.set_ylim((0.49, 0.71))
                minor_ticks = np.linspace(0.50, 0.70, 11)
                major_ticks = np.linspace(0.50, 0.70, 6)
            elif n_meas == 4:
                ax.set_ylim((-0.05, 0.65))
                minor_ticks = np.linspace(0.00, 0.65, 11)
                major_ticks = np.linspace(0.00, 0.65, 6)
            elif n_meas == 5:
                ax.set_ylim((-0.05, 0.45))
                minor_ticks = np.linspace(0.00, 0.45, 11)
                major_ticks = np.linspace(0.00, 0.45, 6)
            elif n_meas == 6:
                ax.set_ylim((-0.05, 1.00))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            elif n_meas == 7:
                ax.set_ylim((-0.05, 1.05))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
            ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/bp_by_res.jpg", bbox_inches="tight")
    plt.clf()
    plt.close(fig)

    # Plots of medians:
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
    axes = axes.flatten()
    for ax, meas in zip(axes, ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):
        ax.plot([5, 10, 25, 50], [x for x, _, _, _, _ in Q2s_by_res[meas]], color=colors[0], linestyle="dashdot")
        ax.plot([5, 10, 25, 50], [x for _, x, _, _, _ in Q2s_by_res[meas]], color=colors[1], linestyle="dashdot")
        ax.plot([5, 10, 25, 50], [x for _, _, x, _, _ in Q2s_by_res[meas]], color=colors[2], linestyle="dashdot")
        ax.plot([5, 10, 25, 50], [x for _, _, _, x, _ in Q2s_by_res[meas]], color=colors[3], linestyle="dashdot")
        ax.plot([5, 10, 25, 50], [x for _, _, _, _, x in Q2s_by_res[meas]], color=colors[4], linestyle="dashdot")
        ax.plot([5, 10, 25, 50], [x for x, _, _, _, _ in Q2s_by_res[meas]], "o", color=colors[0])
        ax.plot([5, 10, 25, 50], [x for _, x, _, _, _ in Q2s_by_res[meas]], "o", color=colors[1])
        ax.plot([5, 10, 25, 50], [x for _, _, x, _, _ in Q2s_by_res[meas]], "o", color=colors[2])
        ax.plot([5, 10, 25, 50], [x for _, _, _, x, _ in Q2s_by_res[meas]], "o", color=colors[3])
        ax.plot([5, 10, 25, 50], [x for _, _, _, _, x in Q2s_by_res[meas]], "o", color=colors[4])
        ax.set_title(meas, fontsize=12)
        ax.grid(color="green", linestyle="--", linewidth=0.5, axis="y")
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.array([5, 10, 25, 50])))
        ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.suptitle("Change in resolution", fontsize=16)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/median_by_res.jpg", bbox_inches="tight")
    plt.show()

    # 2) CHANGE IN CONTACT DENSITY

    # Storing the medians for plots!
    Q2s_by_cd = dict()
    for meas in ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]:
        Q2s_by_cd[meas] = []

    # Looping in contact density:
    fig, axes = plt.subplots(8, 4, figsize=(6, 9))
    for n_cd, cd in enumerate(contact_densities):

        # Extract rows referring to current contact density:
        sliced_results = results.loc[results["Contact Density"] == cd]

        # Medians for current contact density:
        Q2s_this_cd = dict()

        for n_meas, clas_meas_name in enumerate(["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):

            # Medians for current resolution and current measure:
            Q2s_this_cd[clas_meas_name] = []
            for method in ["stripepy1", "stripepy2", "chromosight", "stripecaller", "stripenn"]:
                Q2s_this_cd[clas_meas_name].append(
                    np.median(sliced_results.loc[sliced_results["Method"] == method][clas_meas_name].values)
                )
            Q2s_by_cd[clas_meas_name].append(Q2s_this_cd[clas_meas_name])

            # Box Plots
            ax = axes[n_meas, n_cd]
            sns.boxplot(
                y=clas_meas_name,
                x="Method",
                data=sliced_results,
                ax=ax,
                palette=colors,
                hue="Method",
                orient="v",
                width=0.5,
                fliersize=3,
            )
            ax.set(xlabel=None, ylabel=None)
            ax.tick_params(labelbottom=False, bottom=False)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.axes.get_xaxis().set_visible(False)
            if n_cd == 0:
                ax.set_ylabel(clas_meas_name, fontsize=12)
            if n_meas == 0:
                ax.set_title(r"$\delta$" f" = {int(cd)}", fontsize=12)
                ax.set_ylim((-0.02, 0.50))
                minor_ticks = np.linspace(0.00, 0.50, 11)
                major_ticks = np.linspace(0.00, 0.50, 6)
            elif n_meas == 1:
                ax.set_ylim((0.90, 1.0001))
                minor_ticks = np.linspace(0.90, 1.00, 11)
                major_ticks = np.linspace(0.90, 1.00, 6)
            elif n_meas == 2:
                ax.set_ylim((-0.02, 1.03))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            elif n_meas == 3:
                ax.set_ylim((0.49, 0.72))
                minor_ticks = np.linspace(0.50, 0.70, 11)
                major_ticks = np.linspace(0.50, 0.70, 6)
            elif n_meas == 4:
                ax.set_ylim((-0.02, 0.70))
                minor_ticks = np.linspace(0.00, 0.70, 11)
                major_ticks = np.linspace(0.00, 0.70, 6)
            elif n_meas == 5:
                ax.set_ylim((-0.02, 0.45))
                minor_ticks = np.linspace(0.00, 0.45, 11)
                major_ticks = np.linspace(0.00, 0.45, 6)
            elif n_meas == 6:
                ax.set_ylim((-0.03, 0.975))
                minor_ticks = np.linspace(0.00, 0.95, 11)
                major_ticks = np.linspace(0.00, 0.95, 6)
            elif n_meas == 7:
                ax.set_ylim((-0.05, 1.02))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
            ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/bp_by_cd.jpg", bbox_inches="tight")
    plt.clf()
    plt.close(fig)

    # Plots of medians:
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
    axes = axes.flatten()
    for ax, meas in zip(axes, ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):
        ax.plot([1, 5, 10, 15], [x for x, _, _, _, _ in Q2s_by_cd[meas]], color=colors[0], linestyle="dashdot")
        ax.plot([1, 5, 10, 15], [x for _, x, _, _, _ in Q2s_by_cd[meas]], color=colors[1], linestyle="dashdot")
        ax.plot([1, 5, 10, 15], [x for _, _, x, _, _ in Q2s_by_cd[meas]], color=colors[2], linestyle="dashdot")
        ax.plot([1, 5, 10, 15], [x for _, _, _, x, _ in Q2s_by_cd[meas]], color=colors[3], linestyle="dashdot")
        ax.plot([1, 5, 10, 15], [x for _, _, _, _, x in Q2s_by_cd[meas]], color=colors[4], linestyle="dashdot")
        ax.plot([1, 5, 10, 15], [x for x, _, _, _, _ in Q2s_by_cd[meas]], "o", color=colors[0])
        ax.plot([1, 5, 10, 15], [x for _, x, _, _, _ in Q2s_by_cd[meas]], "o", color=colors[1])
        ax.plot([1, 5, 10, 15], [x for _, _, x, _, _ in Q2s_by_cd[meas]], "o", color=colors[2])
        ax.plot([1, 5, 10, 15], [x for _, _, _, x, _ in Q2s_by_cd[meas]], "o", color=colors[3])
        ax.plot([1, 5, 10, 15], [x for _, _, _, _, x in Q2s_by_cd[meas]], "o", color=colors[4])
        ax.set_title(meas, fontsize=12)
        ax.grid(color="green", linestyle="--", linewidth=0.5, axis="y")
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.array([1, 5, 10, 15])))
        ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.suptitle("Change in contact density", fontsize=16)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/median_by_cd.jpg", bbox_inches="tight")
    plt.show()

    # 3) CHANGE IN NOISE LEVELS

    # Storing the medians for plots!
    Q2s_by_ns = dict()
    for meas in ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]:
        Q2s_by_ns[meas] = []

    # Looping in noise levels:
    fig, axes = plt.subplots(8, 4, figsize=(6, 9))

    for n_ns, noise in enumerate(noises):

        # Extract rows referring to current noise:
        sliced_results = results.loc[results["Noise"] == noise]

        # Medians for current contact density:
        Q2s_this_ns = dict()

        for n_meas, clas_meas_name in enumerate(["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):

            # Medians for current resolution and current measure:
            Q2s_this_ns[clas_meas_name] = []
            for method in ["stripepy1", "stripepy2", "chromosight", "stripecaller", "stripenn"]:
                Q2s_this_ns[clas_meas_name].append(
                    np.median(sliced_results.loc[sliced_results["Method"] == method][clas_meas_name].values)
                )
            Q2s_by_ns[clas_meas_name].append(Q2s_this_ns[clas_meas_name])

            # Box Plots
            ax = axes[n_meas, n_ns]
            sns.boxplot(
                y=clas_meas_name,
                x="Method",
                data=sliced_results,
                ax=ax,
                palette=colors,
                hue="Method",
                orient="v",
                width=0.5,
                fliersize=3,
            )
            ax.set(xlabel=None, ylabel=None)
            ax.tick_params(labelbottom=False, bottom=False)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.axes.get_xaxis().set_visible(False)
            if n_ns == 0:
                ax.set_ylabel(clas_meas_name, fontsize=12)
            if n_meas == 0:
                ax.set_title(r"$\sigma$" f" = {int(noise / 1000)}k", fontsize=12)
                ax.set_ylim((-0.02, 0.500))
                minor_ticks = np.linspace(0.00, 0.500, 11)
                major_ticks = np.linspace(0.00, 0.500, 6)
            elif n_meas == 1:
                ax.set_ylim((0.90, 1.01))
                minor_ticks = np.linspace(0.90, 1.00, 11)
                major_ticks = np.linspace(0.90, 1.00, 6)
            elif n_meas == 2:
                ax.set_ylim((-0.02, 1.02))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            elif n_meas == 3:
                ax.set_ylim((0.49, 0.72))
                minor_ticks = np.linspace(0.50, 0.70, 11)
                major_ticks = np.linspace(0.50, 0.70, 6)
            elif n_meas == 4:
                ax.set_ylim((-0.02, 0.70))
                minor_ticks = np.linspace(0.00, 0.70, 11)
                major_ticks = np.linspace(0.00, 0.70, 6)
            elif n_meas == 5:
                ax.set_ylim((-0.02, 0.45))
                minor_ticks = np.linspace(0.00, 0.45, 11)
                major_ticks = np.linspace(0.00, 0.45, 6)
            elif n_meas == 6:
                ax.set_ylim((-0.03, 1.00))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            elif n_meas == 7:
                ax.set_ylim((-0.05, 1.05))
                minor_ticks = np.linspace(0.00, 1.00, 11)
                major_ticks = np.linspace(0.00, 1.00, 6)
            ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
            ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/bp_by_ns.jpg", bbox_inches="tight")
    plt.clf()
    plt.close(fig)

    # Plots of medians:
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
    axes = axes.flatten()
    for ax, meas in zip(axes, ["TPR", "TNR", "PPV", "bACC", "GM", "MCC", "AHR", "FGC"]):
        ax.plot([5, 10, 15, 20], [x for x, _, _, _, _ in Q2s_by_ns[meas]], color=colors[0], linestyle="dashdot")
        ax.plot([5, 10, 15, 20], [x for _, x, _, _, _ in Q2s_by_ns[meas]], color=colors[1], linestyle="dashdot")
        ax.plot([5, 10, 15, 20], [x for _, _, x, _, _ in Q2s_by_ns[meas]], color=colors[2], linestyle="dashdot")
        ax.plot([5, 10, 15, 20], [x for _, _, _, x, _ in Q2s_by_ns[meas]], color=colors[3], linestyle="dashdot")
        ax.plot([5, 10, 15, 20], [x for _, _, _, _, x in Q2s_by_ns[meas]], color=colors[4], linestyle="dashdot")
        ax.plot([5, 10, 15, 20], [x for x, _, _, _, _ in Q2s_by_ns[meas]], "o", color=colors[0])
        ax.plot([5, 10, 15, 20], [x for _, x, _, _, _ in Q2s_by_ns[meas]], "o", color=colors[1])
        ax.plot([5, 10, 15, 20], [x for _, _, x, _, _ in Q2s_by_ns[meas]], "o", color=colors[2])
        ax.plot([5, 10, 15, 20], [x for _, _, _, x, _ in Q2s_by_ns[meas]], "o", color=colors[3])
        ax.plot([5, 10, 15, 20], [x for _, _, _, _, x in Q2s_by_ns[meas]], "o", color=colors[4])
        ax.set_title(meas, fontsize=12)
        ax.grid(color="green", linestyle="--", linewidth=0.5, axis="y")
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.array([5, 10, 15, 20])))
        ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.suptitle("Change in noise level", fontsize=16)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/median_by_ns.jpg", bbox_inches="tight")
    plt.show()


def global_boxplots(results, path2output):
    """
    Plot boxplots that describe the methods' performance globally (all 16 Hi-C matrices at 4 different resolutions)
    :param results: pandas dataframe containing all measures
    :param path2output: path to folder where to save output jpg images
    :return: -
    """

    # GLOBAL BOXPLOTS
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
    axes = axes.flatten()
    for ax, clas_meas_name in zip(axes, ["TPR", "TNR", "PPV", "AHR", "bACC", "GM", "MCC", "FGC"]):
        sns.boxplot(
            y=clas_meas_name,
            x="Method",
            data=results,
            ax=ax,
            palette=colors,
            hue="Method",
            orient="v",
            width=0.5,
            fliersize=3,
        )
        ax.set(xlabel=None, ylabel=None)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.set_yticks([0.05 * i for i in range(21)], minor=True)
        ax.set_yticks([0.2 * i for i in range(6)], minor=False)
        ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
        ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)
        ax.yaxis.set_tick_params(labelsize="medium")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_title(clas_meas_name, fontsize=12)
        ax.set_ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(f"{path2output}boxplots/bp.jpg", bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def heatmaps(results, resolutions, contact_densities, noises, path2output):
    """
    Plot heatmaps that compare anchor points found or not found by pair of methods, with respect of both classification
    and prediction.
    :param results: pandas dataframe containing all measures
    :param resolutions: resolutions in the benchmark
    :param contact_densities: contact densities in the benchmark
    :param noises: noise levels in the benchmark
    :param path2output: path to folder where to save output jpg images
    :return: -
    """

    # Retrieve results of each method:
    M1 = results.loc[results["Method"] == "stripepy1"]
    M2 = results.loc[results["Method"] == "stripepy2"]
    M3 = results.loc[results["Method"] == "chromosight"]
    M4 = results.loc[results["Method"] == "stripecaller"]
    M5 = results.loc[results["Method"] == "stripenn"]

    # RECOGNITION

    for key in ["is_anchor_found", "classification_vector"]:

        # CFR[i][j] contains the 2x2 heatmap that compares methods f"M{i+1}" to f"M{j+2}"... the notation here is
        # debatable... In short, it means that CFR[0,0] compares M1 to M3
        CFR = np.zeros((2, 3, 2, 2))

        # Generate all combinations of resolutions, contact densities, and noise levels
        combinations = itertools.product(resolutions, contact_densities, noises)

        # Loop over combinations:
        for resolution, contact_density, noise in combinations:

            # Update each entry of CFR:
            if key == "is_anchor_found":

                # Retrieve all entries in M1,...,M5 with key "is_anchor_found"
                V = []
                for M in [M1, M2, M3, M4, M5]:
                    v = M.loc[
                        (M["Resolution"] == resolution)
                        & (M["Contact Density"] == contact_density)
                        & (M["Noise"] == noise)
                    ][f"{key}"].values[0]
                    V.append(v)
                V = np.array(V)

                # Analyse the four possible scenarios:
                for i, j in np.ndindex(2, 3):
                    CFR[i][j] += np.array(
                        [
                            [
                                np.sum((V[i] == 0) & (V[j + 2] == 0)) / len(V[i]) * 100,
                                np.sum((V[i] == 1) & (V[j + 2] == 0)) / len(V[i]) * 100,
                            ],
                            [
                                np.sum((V[i] == 0) & (V[j + 2] == 1)) / len(V[i]) * 100,
                                np.sum((V[i] == 1) & (V[j + 2] == 1)) / len(V[i]) * 100,
                            ],
                        ]
                    )

            else:

                # Retrieve all entries in M1,...,M5 with key "classification_vector"
                V = []
                for M in [M1, M2, M3, M4, M5]:
                    v = M.loc[
                        (M["Resolution"] == resolution)
                        & (M["Contact Density"] == contact_density)
                        & (M["Noise"] == noise)
                    ][f"{key}"].values[0]
                    V.append(v)
                V = np.array(V)

                # Retrieve ground truth classification:
                GT = np.array(
                    M1.loc[
                        (M1["Resolution"] == resolution)
                        & (M1["Contact Density"] == contact_density)
                        & (M1["Noise"] == noise)
                    ][f"GT_{key}"].values[0]
                )
                is_anchor = np.where(GT == 1)[0]

                # Analyse the four possible scenarios:
                for i, j in np.ndindex(2, 3):
                    CFR[i][j] += np.array(
                        [
                            [
                                np.sum((V[i][is_anchor] == 0) & (V[j + 2][is_anchor] == 0)) / len(is_anchor) * 100,
                                np.sum((V[i][is_anchor] == 1) & (V[j + 2][is_anchor] == 0)) / len(is_anchor) * 100,
                            ],
                            [
                                np.sum((V[i][is_anchor] == 0) & (V[j + 2][is_anchor] == 1)) / len(is_anchor) * 100,
                                np.sum((V[i][is_anchor] == 1) & (V[j + 2][is_anchor] == 1)) / len(is_anchor) * 100,
                            ],
                        ]
                    )

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        categories = ["Not Identified", "Identified"]

        for num_CFR, (CFR, ax) in enumerate(
            zip([CFR[0][0], CFR[0][1], CFR[0][2], CFR[1][0], CFR[1][1], CFR[1][2]], axes)
        ):
            sns.heatmap(
                CFR.transpose() / (len(resolutions) * len(contact_densities) * len(noises)),
                annot=True,
                ax=ax,
                cmap="Blues",
                xticklabels=categories,
                yticklabels=categories,
                cbar=False,
                vmin=0,
                vmax=100,
                fmt=".1f",
                annot_kws={"size": 12},
            )
            for t in ax.texts:
                t.set_text(t.get_text() + " %")

            # x- and y-labels:
            if num_CFR == 0:
                ax.set_ylabel("stripepy 1", fontsize=14)
                ax.set_xlabel("chromosight", fontsize=14)
            if num_CFR == 1:
                ax.set_xlabel("stripecaller", fontsize=14)
            if num_CFR == 2:
                ax.set_xlabel("stripenn", fontsize=14)
            if num_CFR == 3:
                ax.set_ylabel("stripepy 2", fontsize=14)
            ax.tick_params(left=False, bottom=False)

            # Reduce the size of xtick labels for each subplot
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.axis("scaled")
            ax.invert_xaxis()
            ax.xaxis.set_label_position("top")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        plt.savefig(f"{path2output}heatmap_{key}.jpg")
        plt.clf()
        plt.close(fig)


def recoverable_anchors_recognition(
    M1, resolutions, contact_densities, noises, base_path, file_name_base, GT_name, probability_threshold
):
    """
    Quantify the percentage of stripepy's anchor points that can be recovered by lowering the threshold of the
    topological persistence:
    :param M1: dictionary containing predictions from stripepy-nf, used to compute the ratio of recoverable anchors
    :param resolutions: resolutions in the benchmark
    :param contact_densities: contact densities in the benchmark
    :param noises: noise levels in the benchmark
    :param base_path: path to the ground truth file
    :param file_name_base: name of the file that was used to create the ground truth (it is grch38_h1_rad21)
    :param GT_name: name of the bed file containing coordinates and occupancy of each extrusion barrier
    :param probability_threshold: cut-off value for the occupancy
    :return: -
    """

    ratio_recoverable_over_undetected = []
    ratio_recoverable_over_all_anchors = []
    persistences_to_analyze = []

    # Generate all combinations of resolutions, contact densities, and noise levels
    combinations = itertools.product(resolutions, contact_densities, noises)

    # Loop over the combinations
    for resolution, contact_density, noise in combinations:

        # Retrieve chromosome names:
        path2GT = f"{base_path}/MoDLE-benchmark/"
        file_name = f"{file_name_base}_{contact_density}_{noise}"
        path2mcool = f"{base_path}/MoDLE-benchmark/data/{file_name}/"
        c = hictkpy.File(f"{path2mcool}{file_name}.mcool::resolutions/{resolution}", resolution)
        c_names = list(c.chromosomes().keys())
        c_ids = list(range(len(c_names)))
        c_pairs = list(zip(c_ids, c_names))

        # Retrieve 0-1 vector that states whether an anchor was found or not. NB: it is NOT the classification vector!
        is_anchor_found = M1.loc[
            (M1["Resolution"] == resolution) & (M1["Contact Density"] == contact_density) & (M1["Noise"] == noise)
        ]["is_anchor_found"].values[0]

        # Number of GT anchors per chromosome and per part of the matrix:
        num_GT_anchors_per_chrom_LT = [0] * len(c_names)
        num_GT_anchors_per_chrom_UT = [0] * len(c_names)

        # Total number of ground-truth anchors per chromosome:
        for i, chr in enumerate(c_names):
            GT = retrieve_ground_truth(path2GT, GT_name, chr, resolution, probability_threshold, False)
            num_GT_anchors_per_chrom_LT[i] = len(GT["L_anchors"])
            num_GT_anchors_per_chrom_UT[i] = len(GT["U_anchors"])

        # Total number of ground-truth anchors -- lower-triangular:
        LT_num_anchors = np.sum(num_GT_anchors_per_chrom_LT)
        LT_num_anchors_undetected = 0
        LT_num_anchors_undetected_but_detectable = 0
        LT_num_anchors_detected = 0

        # Total number of ground-truth anchors -- upper-triangular:
        UT_num_anchors = np.sum(num_GT_anchors_per_chrom_UT)
        UT_num_anchors_undetected = 0
        UT_num_anchors_undetected_but_detectable = 0
        UT_num_anchors_detected = 0

        # Indices to slice is_anchor_found and retrieve lower- and upper-triangular parts:
        LT_start = 0
        UT_start = np.sum(num_GT_anchors_per_chrom_LT)

        for i, chr in enumerate(c_names):

            # Retrieve ground truth:
            GT = retrieve_ground_truth(path2GT, GT_name, chr, resolution, probability_threshold, False)

            # Slice 0-1 vectors:
            LT_is_anchor_found_loc = is_anchor_found[LT_start : LT_start + num_GT_anchors_per_chrom_LT[i]]
            UT_is_anchor_found_loc = is_anchor_found[UT_start : UT_start + num_GT_anchors_per_chrom_UT[i]]

            # Update indices to slice is_anchor_found and retrieve lower- and upper-triangular parts:
            LT_start += num_GT_anchors_per_chrom_LT[i]
            UT_start += num_GT_anchors_per_chrom_UT[i]

            # Retrieve topological persistence values:
            path2persistence = f"output/MoDLE-benchmark/stripepy/{file_name}/{resolution}/{chr}/global/all"
            LT_MPs = np.loadtxt(f"{path2persistence}/LT_MPs.txt", dtype=int).tolist()
            UT_MPs = np.loadtxt(f"{path2persistence}/UT_MPs.txt", dtype=int).tolist()
            LT_pers_of_MPs = np.loadtxt(f"{path2persistence}/LT_pers_of_MPs.txt").tolist()
            UT_pers_of_MPs = np.loadtxt(f"{path2persistence}/UT_pers_of_MPs.txt").tolist()

            # How many anchors have already been detected?
            LT_num_anchors_detected += np.sum(LT_is_anchor_found_loc)
            UT_num_anchors_detected += np.sum(UT_is_anchor_found_loc)

            # How many anchors can be detected?
            for n_e1, e1 in enumerate(LT_is_anchor_found_loc):
                if e1 == 0:
                    LT_num_anchors_undetected += 1
                    missed_anchor = int(GT["L_anchors"][n_e1])
                    if missed_anchor in LT_MPs:
                        LT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [LT_pers_of_MPs[LT_MPs.index(missed_anchor)]]
                    elif missed_anchor - 1 in LT_MPs:
                        LT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [LT_pers_of_MPs[LT_MPs.index(missed_anchor - 1)]]
                    elif missed_anchor + 1 in LT_MPs:
                        LT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [LT_pers_of_MPs[LT_MPs.index(missed_anchor + 1)]]
            for n_e1, e1 in enumerate(UT_is_anchor_found_loc):
                if e1 == 0:
                    UT_num_anchors_undetected += 1
                    missed_anchor = int(GT["U_anchors"][n_e1])
                    if missed_anchor in UT_MPs:
                        UT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [UT_pers_of_MPs[UT_MPs.index(missed_anchor)]]
                    elif missed_anchor - 1 in UT_MPs:
                        UT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [UT_pers_of_MPs[UT_MPs.index(missed_anchor - 1)]]
                    elif missed_anchor + 1 in UT_MPs:
                        UT_num_anchors_undetected_but_detectable += 1
                        persistences_to_analyze += [UT_pers_of_MPs[UT_MPs.index(missed_anchor + 1)]]

        # Computing the percentages:
        num_anchors = LT_num_anchors + UT_num_anchors
        # num_anchors_detected = LT_num_anchors_detected + UT_num_anchors_detected
        num_anchors_undetected_but_detectable = (
            LT_num_anchors_undetected_but_detectable + UT_num_anchors_undetected_but_detectable
        )
        num_anchors_undetected = LT_num_anchors_undetected + UT_num_anchors_undetected
        ratio_recoverable_over_undetected += [num_anchors_undetected_but_detectable / num_anchors_undetected]
        ratio_recoverable_over_all_anchors += [num_anchors_undetected_but_detectable / num_anchors]

    print("---ratio_recoverable_over_undetected---")
    df = pd.DataFrame({"ratio_recoverable_over_undetected": np.array(ratio_recoverable_over_undetected) * 100})
    quartiles = df.quantile([0.00, 0.25, 0.5, 0.75, 1.00])
    print("Quartiles:", quartiles)

    print("---ratio_recoverable_over_all_anchors---")
    df = pd.DataFrame({"ratio_recoverable_over_all_anchors": np.array(ratio_recoverable_over_all_anchors) * 100})
    quartiles = df.quantile([0.00, 0.25, 0.5, 0.75, 1.00])
    print("Quartiles:", quartiles)


def recoverable_anchors_classification(
    M1, resolutions, contact_densities, noises, base_path, file_name_base, GT_name, probability_threshold
):
    """
    Quantify the percentage of stripepy's anchor points that can be recovered by lowering the threshold of the
    topological persistence:
    :param M1: dictionary containing predictions from stripepy-nf, used to compute the ratio of recoverable anchors
    :param resolutions: resolutions in the benchmark
    :param contact_densities: contact densities in the benchmark
    :param noises: noise levels in the benchmark
    :param base_path: path to the ground truth file
    :param file_name_base: name of the file that was used to create the ground truth (it is grch38_h1_rad21)
    :param GT_name: name of the bed file containing coordinates and occupancy of each extrusion barrier
    :param probability_threshold: cut-off value for the occupancy
    :return: -
    """

    ratio_recoverable_over_undetected = []
    ratio_recoverable_over_all_anchors = []
    persistences_to_analyze = []

    # Generate all combinations of resolutions, contact densities, and noise levels
    combinations = itertools.product(resolutions, contact_densities, noises)

    # Loop over the combinations
    for resolution, contact_density, noise in combinations:

        # Retrieve chromosome names:
        path2GT = f"{base_path}/MoDLE-benchmark/"
        file_name = f"{file_name_base}_{contact_density}_{noise}"
        path2mcool = f"{base_path}/MoDLE-benchmark/data/{file_name}/"
        c = hictkpy.File(f"{path2mcool}{file_name}.mcool::resolutions/{resolution}", resolution)
        c_names = list(c.chromosomes().keys())
        c_ids = list(range(len(c_names)))
        c_pairs = list(zip(c_ids, c_names))

        # Retrieve 0-1 vector that states whether an anchor was found or not:
        pred_clas_vec = M1.loc[
            (M1["Resolution"] == resolution) & (M1["Contact Density"] == contact_density) & (M1["Noise"] == noise)
        ]["classification_vector"].values[0]

        # Retrieve ground truth classification:
        GT_clas_vec = np.array(
            M1.loc[
                (M1["Resolution"] == resolution) & (M1["Contact Density"] == contact_density) & (M1["Noise"] == noise)
            ]["GT_classification_vector"].values[0]
        )

        # Classification vectors obtained when lowering the admitting maxima points with lower topological persistence:
        LT_new_clas_vector = []
        UT_new_clas_vector = []

        # Perform the check chromosome per chromosome:
        for i, chr in enumerate(c_names):

            # Number of bins:
            bins = c.bins()[:]
            n_bins = len(bins[bins["chrom"] == chr])

            # Retrieve maxima points and update the new classification vectors:
            path2persistence = f"output/MoDLE-benchmark/stripepy/{file_name}/{resolution}/{chr}/global/all"
            LT_MPs = np.loadtxt(f"{path2persistence}/LT_MPs.txt", dtype=int).tolist()
            UT_MPs = np.loadtxt(f"{path2persistence}/UT_MPs.txt", dtype=int).tolist()
            LT_new_clas_vector += np.isin(np.arange(n_bins), LT_MPs).astype(int).tolist()
            UT_new_clas_vector += np.isin(np.arange(n_bins), UT_MPs).astype(int).tolist()

        # Merge new classification vectors for lower- and upper- triangular parts:
        new_clas_vector = np.array(LT_new_clas_vector + UT_new_clas_vector)

        # Computing the percentages:
        num_lost_anchors = np.sum((np.array(pred_clas_vec) == 0) & (np.array(GT_clas_vec) == 1))
        num_lost_but_recoverable_anchors = np.sum(
            (np.array(new_clas_vector) == 1) & (np.array(pred_clas_vec) == 0) & (np.array(GT_clas_vec) == 1)
        )
        ratio_recoverable_over_undetected.append(num_lost_but_recoverable_anchors / num_lost_anchors)
        ratio_recoverable_over_all_anchors.append(num_lost_but_recoverable_anchors / np.sum(np.array(GT_clas_vec) == 1))

    print("---ratio_recoverable_over_undetected---")
    df = pd.DataFrame({"ratio_recoverable_over_undetected": np.array(ratio_recoverable_over_undetected) * 100})
    quartiles = df.quantile([0.00, 0.25, 0.5, 0.75, 1.00])
    print("Quartiles:", quartiles)

    print("---ratio_recoverable_over_all_anchors---")
    df = pd.DataFrame({"ratio_recoverable_over_all_anchors": np.array(ratio_recoverable_over_all_anchors) * 100})
    quartiles = df.quantile([0.00, 0.25, 0.5, 0.75, 1.00])
    print("Quartiles:", quartiles)
