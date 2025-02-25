#!/usr/bin/env python3

# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import json
import logging
import pathlib
import re
import textwrap
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd

from stripepy.data_structures import Result
from stripepy.io import ResultFile


def make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()

    cli.add_argument(
        "result-hdf5",
        type=pathlib.Path,
        help="Path to a HDF5 file generated by stripepy call.",
    )
    cli.add_argument(
        "output-dir",
        type=pathlib.Path,
        help="Path to the output directory.",
    )
    cli.add_argument(
        "--output-prefix",
        type=str,
        help="File name prefix used to generate output file names. When not provided, the output prefix is inferred from the input file name.",
    )
    cli.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite existing file(s).",
    )

    return cli


def hash_file(path: pathlib.Path, chunk_size: int = 16 << 20) -> str:
    logging.info("hashing file %s...", path)
    with path.open("rb") as f:
        hasher = hashlib.sha256()
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return hasher.hexdigest()
            hasher.update(chunk)


def extract_metadata(path: pathlib.Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return dict(h5.attrs)


def write_json(data: Dict[str, Any], path: pathlib.Path, force: bool):
    logging.info('writing JSON to file "%s"...', path)
    if path.exists() and not force:
        raise RuntimeError(f'Refusing to overwrite file "{path}". Pass --force to overwrite.')

    for k, v in data.items():
        if isinstance(v, np.integer):
            data[k] = int(v)
        elif isinstance(v, np.floating):
            data[k] = float(v)
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()

        if k == "metadata":
            try:
                data[k] = json.loads(v)
            except Exception:  # noqa
                pass

    with path.open("w") as f:
        json.dump(data, f, indent=2)  # noqa


def to_parquet(df: pd.DataFrame, path: pathlib.Path, force: bool):
    logging.info('writing parquet to file "%s"...', path)
    if path.exists() and not force:
        raise RuntimeError(f'Refusing to overwrite file "{path}". Pass --force to overwrite.')

    df.to_parquet(path, compression="zstd", engine="pyarrow", compression_level=19)


def process_chromosome(res: Result, output_prefix: pathlib.Path, force: bool):
    logging.info('processing chromosome "%s"...', res.chrom[0])
    output_prefix /= res.chrom[0]
    output_prefix.mkdir(exist_ok=True)
    attrs = {
        "min_persistence": res.min_persistence,
        "roi": res.roi,
    }

    write_json(attrs, output_prefix / "attributes.json", force)

    fields = (
        "all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_minimum_points",
        "persistence_of_all_maximum_points",
        "persistent_minimum_points",
        "persistent_maximum_points",
        "persistence_of_minimum_points",
        "persistence_of_maximum_points",
        "pseudodistribution",
    )

    for location in ("LT", "UT"):
        to_parquet(
            res.get_stripe_bio_descriptors(location),
            output_prefix / f"bio_descriptors_{location.lower()}.parquet",
            force,
        )
        to_parquet(
            res.get_stripe_geo_descriptors(location),
            output_prefix / f"geo_descriptors_{location.lower()}.parquet",
            force,
        )

        for field in fields:
            dest = output_prefix / f"{field}_{location.lower()}.parquet"
            try:
                to_parquet(
                    pd.DataFrame({field: res.get(field, location)}),
                    dest,
                    force,
                )
            except RuntimeError as e:
                if not re.match(r"Attribute .* is not set", str(e)):
                    raise

                with dest.with_suffix(".missing").open("w") as f:
                    f.write(f"{e}\n")


def write_readme(result_file: pathlib.Path, output_prefix: pathlib.Path, chroms: List[str], force: bool):
    readme = output_prefix / "README.md"
    if readme.exists() and not force:
        raise RuntimeError(f'Refusing to overwrite file "{readme}". Pass --force to overwrite.')

    with readme.open("w") as f:
        chrom1 = chroms[0]
        if len(chroms) > 1:
            chrom2 = chroms[1]
        else:
            chrom2 = "abc"

        f.write(
            textwrap.dedent(
                f"""
        # README.md

        This folder contains the tables and metadata extracted from file \"{result_file.name}\".

        The folder has the following structure:

        ```tree
        {output_prefix.stem}
        ├── attributes.json
        ├── checksum.sha256
        ├── {chrom1}
        │   ├── all_maximum_points_lt.parquet
        │   ├── all_maximum_points_ut.parquet
        │   ├── all_minimum_points_lt.parquet
        │   ├── all_minimum_points_ut.parquet
        │   ├── attributes.json
        │   ├── bio_descriptors_lt.parquet
        │   ├── bio_descriptors_ut.parquet
        │   ├── geo_descriptors_lt.parquet
        │   ├── geo_descriptors_ut.parquet
        │   ├── persistence_of_all_maximum_points_lt.parquet
        │   ├── persistence_of_all_maximum_points_ut.parquet
        │   ├── persistence_of_all_minimum_points_lt.parquet
        │   ├── persistence_of_all_minimum_points_ut.parquet
        │   ├── persistence_of_maximum_points_lt.missing
        │   ├── persistence_of_maximum_points_ut.missing
        │   ├── persistence_of_minimum_points_lt.missing
        │   ├── persistence_of_minimum_points_ut.missing
        │   ├── persistent_maximum_points_lt.missing
        │   ├── persistent_maximum_points_ut.missing
        │   ├── persistent_minimum_points_lt.missing
        │   ├── persistent_minimum_points_ut.missing
        │   ├── pseudodistribution_lt.parquet
        │   └── pseudodistribution_ut.parquet
        ├── {chrom2}
        ...
        ```

        Explanation:
         - `attributes.json`: JSON file with the attributes stored in the root of the result file.
         - `checksum.sha256`: checksum of the HDF5 file from which the data comes from.
         - `{chrom1}`: folder containing all data for a chromosome named `{chrom1}`.
           This folder contains:
           - `attributes.json`: JSON file with the attributes for the current chromosome.
           - `xxx_where.parquet`: PARQUET file with the data from table "xxx" for location "where" from the current chromosome.
           - `xxx_where.missing`: text file with the exception message raised when trying to access tables without data.
        """
            ).lstrip("\n"),
        )


def setup_logger(level: str):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().setLevel(level)


def main():
    setup_logger("INFO")
    args = vars(make_cli().parse_args())

    output_dir = pathlib.Path(args["output-dir"])

    result_file = args["result-hdf5"]

    output_prefix = args["output_prefix"]
    if output_prefix is None:
        output_prefix = result_file.stem

    output_prefix = output_dir / output_prefix
    output_prefix.mkdir(exist_ok=True, parents=True)

    checksum_file = output_prefix / "checksum.sha256"
    if checksum_file.exists() and not args["force"]:
        raise RuntimeError(f'Refusing to overwrite file "{checksum_file}". Pass --force to overwrite.')

    with checksum_file.open("w") as f:
        f.write(f"{hash_file(result_file)}  {result_file.stem}\n")

    attrs = extract_metadata(result_file)
    write_json(attrs, output_prefix / "attributes.json", args["force"])

    with ResultFile(result_file) as h5:
        chroms = list(sorted(h5.chromosomes))
        for chrom in chroms:
            process_chromosome(h5[chrom], output_prefix, args["force"])

    write_readme(result_file, output_prefix, chroms, args["force"])


if __name__ == "__main__":
    main()
