#!/usr/bin/env python3

# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import logging
import multiprocessing as mp
import pathlib
import shutil
import subprocess as sp
import tempfile
from typing import Dict, List, Union

import hictkpy as htk


def existing_file(arg: str) -> pathlib.Path:
    if (path := pathlib.Path(arg)).is_file():
        return path

    raise FileNotFoundError(arg)


def positive_int(arg) -> int:
    if (n := int(arg)) > 0:
        return n

    raise ValueError("Not a positive int")


def num_cpus(arg: str) -> int:
    try:
        n = int(arg)
        if 0 < n <= mp.cpu_count():
            return n
    except:  # noqa
        pass

    raise ValueError(f"Not a valid number of CPU cores (allowed values are integers between 1 and {mp.cpu_count()})")


def make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        "Given a matrix in .mcool or .hic format, generate a matrix file suitable for testing."
    )

    cli.add_argument("input-matrix", type=existing_file, help="Path to a Hi-C matrix in .cool, .mcool or .hic format.")
    cli.add_argument("output-matrix", type=pathlib.Path, help="Path where to store the resulting output matrix file.")
    cli.add_argument(
        "--base-resolution",
        type=positive_int,
        help="Base resolution used to generate the test file. Required when file is in .mcool or .hic format.",
    )
    cli.add_argument(
        "--additional-resolutions",
        nargs="*",
        type=positive_int,
        help="One or more resolutions to be generated. Should be multiples of the base resolution.",
    )
    cli.add_argument(
        "--hictk-bin",
        type=existing_file,
        help="Path to the hictk binary. When not provided, hictk's binary is looked up in the system PATH.",
    )
    cli.add_argument(
        "--genomic-belt",
        type=int,
        default=5_000_000,
        help="Radius of the band, centred around the diagonal. Interactions outside of this band are ignored.",
    )
    cli.add_argument(
        "-p",
        "--nproc",
        type=num_cpus,
        default=1,
        help="Maximum number of parallel processes to use. Only used when processing .hic files.",
    )

    cli.add_argument("--force", action="store_true", default=False, help="Overwrite existing file(s).")

    return cli


def find_hictk(path: Union[pathlib.Path, None]) -> pathlib.Path:
    if path is None:
        path = "hictk"

    if shutil.which(path):
        return path

    raise FileNotFoundError("unable to locate hictk's binary")


def dump_chroms(chroms: Dict[str, int], tmpdir: pathlib.Path) -> pathlib.Path:
    path = tmpdir / "chroms.sizes"
    with open(path, "w") as f:
        for chrom, size in chroms.items():
            f.write(f"{chrom}\t{size}\n")

    return path


def run_hictk_load(
    f: htk.File, genomic_belt: int, tmpdir: pathlib.Path, threads: int, skip_all_vs_all_matrix: bool
) -> pathlib.Path:
    if f.is_hic():
        path = tmpdir / "matrix.hic"
        w = htk.hic.FileWriter(
            path,
            f.chromosomes(),
            f.resolution(),
            f.attributes()["assembly"],
            tmpdir=tmpdir,
            compression_lvl=12,
            skip_all_vs_all_matrix=skip_all_vs_all_matrix,
            n_threads=threads,
        )
    else:
        path = tmpdir / "matrix.cool"
        w = htk.cooler.FileWriter(
            path, f.chromosomes(), f.resolution(), f.attributes()["assembly"], tmpdir=tmpdir, compression_lvl=9
        )

    genomic_belt = (genomic_belt - f.resolution() - 1) // f.resolution()

    for chrom in f.chromosomes():
        logging.info("fetching interactions for %s...", chrom)
        df = f.fetch(chrom).to_pandas()
        num_interactions = len(df)
        df = df[df["bin2_id"] - df["bin1_id"] < genomic_belt]

        logging.info("dropped %d out of %d interactions", num_interactions - len(df), num_interactions)
        w.add_pixels(df)

    w.finalize(log_lvl="INFO")

    return path


def run_hictk_zoomify(
    hictk: pathlib.Path,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    resolutions: List[int],
    tmpdir: pathlib.Path,
    threads: int,
    force: bool,
):
    resolutions = set(map(str, resolutions))
    resolutions.add(str(htk.File(input_path).resolution()))
    resolutions = list(sorted(resolutions))

    cmd = [
        hictk,
        "zoomify",
        input_path,
        output_path,
        "--resolutions",
        *resolutions,
        "--tmpdir",
        tmpdir,
        "--threads",
        str(threads),
    ]
    if force:
        cmd.append("--force")

    logging.info("running %s", " ".join(str(x) for x in cmd))
    sp.check_call(cmd)


def main():
    args = vars(make_cli().parse_args())

    if args["output-matrix"].exists() and not args["force"]:
        raise RuntimeError(f"refusing to overwrite file {args['output-matrix']}. Pass --force to overwrite.")

    need_to_zoomify = len(args["additional_resolutions"]) > 0

    if need_to_zoomify:
        hictk = find_hictk(args["hictk_bin"])

    if args["base_resolution"] is None and not htk.is_cooler(args["input-matrix"]):
        raise RuntimeError("--base-resolution is required when input matrix is not in .cool format.")

    f = htk.File(args["input-matrix"], resolution=args["base_resolution"])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        single_res_file = run_hictk_load(
            f, args["genomic_belt"], tmpdir, args["nproc"], skip_all_vs_all_matrix=need_to_zoomify
        )

        if not need_to_zoomify:
            single_res_file.rename(args["output-matrix"])
            return

        run_hictk_zoomify(
            hictk,
            single_res_file,
            args["output-matrix"],
            args["additional_resolutions"],
            tmpdir,
            args["nproc"],
            args["force"],
        )


def setup_logger(level: str):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().setLevel(level)


if __name__ == "__main__":
    setup_logger("INFO")
    main()
