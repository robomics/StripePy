#!/usr/bin/env python3

# Copyright (c) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import argparse
import pathlib
import re
import shutil
import subprocess as sp
import textwrap
from typing import Tuple


def make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()

    def valid_executable(s: str) -> pathlib.Path:
        if shutil.which(s):
            return pathlib.Path(s)

        if s == "stripepy":
            raise argparse.ArgumentTypeError("Unable to find stripepy in your PATH.")

        raise argparse.ArgumentTypeError(f'"{s}" is not a valid executable.')

    cli.add_argument(
        "--stripepy",
        type=valid_executable,
        default=pathlib.Path("stripepy"),
        required=False,
        help="Path to stripepy's executable.",
    )

    return cli


def generate_main_header():
    header = """
    ..
      Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
      SPDX-License-Identifier: MIT

    CLI Reference
    #############

    For an up-to-date list of subcommands and CLI options refer to ``stripepy --help``.

    .. _stripepy_help:

    Subcommands
    -----------

    .. code-block:: text

    """

    print(textwrap.dedent(header))


def generate_subcommand_header(subcommand: Tuple[str]):
    subcommand = "stripepy " + " ".join(subcommand)
    bookmark = f'.. _{re.sub(r'\W+', '_', subcommand)}_help:'
    separator = "-" * len(subcommand)
    header = f"""

    {bookmark}

    {subcommand}
    {separator}

    .. code-block:: text
    """

    print(textwrap.dedent(header))


def sanitize(msg: str, stripepy: pathlib.Path) -> str:
    msg = msg.replace(str(stripepy), "stripepy")
    msg = re.sub(r"^\s+$", "", msg, flags=re.MULTILINE)
    return re.sub(r"\s+$", "", msg, flags=re.MULTILINE)


def generate_main_help_msg(stripepy: pathlib.Path):
    msg = sp.check_output([stripepy, "--help"]).decode("utf-8")
    msg = sanitize(msg, stripepy)
    print(textwrap.indent(msg, "  "))


def generate_subcommand_help_msg(stripepy: pathlib.Path, subcommand: Tuple[str]):
    msg = sp.check_output([stripepy, *subcommand, "--help"]).decode("utf-8")
    msg = sanitize(msg, stripepy)
    print(textwrap.indent(msg, "  "))


def main():
    args = vars(make_cli().parse_args())

    stripepy = args["stripepy"]
    if not shutil.which(stripepy):
        if stripepy == "stripepy":
            raise RuntimeError("Unable to find stripepy in your PATH.")

        raise RuntimeError(f'"{stripepy}" is not a valid executable.')

    subcommands = (
        ("call",),
        ("download",),
        ("plot",),
        ("plot", "contact-map"),
        ("plot", "pseudodistribution"),
        ("plot", "stripe-hist"),
        ("view",),
    )

    generate_main_header()
    generate_main_help_msg(stripepy)

    for subcommand in subcommands:
        generate_subcommand_header(subcommand)
        generate_subcommand_help_msg(stripepy, subcommand)


if __name__ == "__main__":
    main()
