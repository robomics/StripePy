#!/usr/bin/env python3

# Copyright (c) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import pathlib
import re
import subprocess as sp


def make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()

    cli.add_argument(
        "--root-dir",
        type=pathlib.Path,
        required=False,
        help="Path to the root of the doc folder.",
    )
    cli.add_argument(
        "--inplace",
        action="store_true",
        default=False,
        help="Do the replacement in-place.",
    )

    return cli


def infer_root_dir(cwd: pathlib.Path | None = None) -> pathlib.Path:
    if cwd is None:
        cwd = pathlib.Path(__file__).parent.resolve()

    res = sp.check_output(["git", "rev-parse", "--show-toplevel"], cwd=cwd).decode("utf-8").split("\n")[0]

    root_dir = pathlib.Path(res)
    if root_dir.is_dir():
        return root_dir

    if cwd == pathlib.Path(__file__).parent.resolve():
        raise RuntimeError("Unable to infer the root of StripePy's repository.")

    return infer_root_dir(pathlib.Path(__file__).parent.resolve())


def subn_checked(pattern: re.Pattern, repl: str, string: str, count: int = 0) -> str:
    string, num_replacements = pattern.subn(repl, string, count)
    if num_replacements == 0:
        raise RuntimeError(f"Failed to match pattern {str(pattern)} to the given string")

    return string


def patch_docs_index_file(path: pathlib.Path, inplace: bool):
    logging.info('Patching "%s"...', path)
    url = os.getenv("READTHEDOCS_CANONICAL_URL")
    if url is None:
        raise RuntimeError("Unable to read url from the READTHEDOCS_CANONICAL_URL env variable")

    logging.info(f'READTHEDOCS_CANONICAL_URL="{url}"')

    toks = url.removeprefix("https://").rstrip("/").split("/")
    if len(toks) < 2:
        raise RuntimeError("Failed to parse READTHEDOCS_CANONICAL_URL variable")

    tgt_domain = toks[0]
    tgt_branch = toks[-1]

    logging.info(f'new_domain="{tgt_domain}"')
    logging.info(f'new_branch="{tgt_branch}"')

    pdf_pattern = re.compile(r"https://stripepy\.readthedocs\.io/_/downloads/en/[\w\-_]+/pdf/")
    html_pattern = re.compile(r"https://stripepy\.readthedocs\.io/en/[\w\-_]+/")

    payload = path.read_text()
    payload = subn_checked(pdf_pattern, f"https://{tgt_domain}/_/downloads/en/{tgt_branch}/pdf/", payload)
    payload = subn_checked(html_pattern, f"https://{tgt_domain}/en/{tgt_branch}/", payload)

    if inplace:
        logging.info(f'Updating file "{path}" inplace...')
        path.write_text(payload)
    else:
        print(payload, end="")


def generate_path_checked(root_dir: pathlib.Path, suffix: pathlib.Path) -> pathlib.Path:
    path = root_dir / suffix
    if not path.exists():
        raise RuntimeError(f'Unable to find file "{suffix}" under {root_dir}')

    return path


def main():
    if "READTHEDOCS" not in os.environ:
        logging.info("Script is not being run by ReadTheDocs. Returning immediately!")
        return

    args = vars(make_cli().parse_args())

    root_dir = args["root_dir"]
    if root_dir is None:
        root_dir = infer_root_dir()

    docs_index_file = generate_path_checked(root_dir, pathlib.Path("docs/index.rst"))
    patch_docs_index_file(docs_index_file, args["inplace"])


if __name__ == "__main__":
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(format=fmt)
    logging.getLogger().setLevel(logging.INFO)
    main()
