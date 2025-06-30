#!/usr/bin/env bash

# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

set -e
set -u
set -o pipefail

ARGC=$#

if [ $ARGC -lt 3 ]; then
  1>&2 echo "Usage:   $0 path/to/result_file_to_parquet.py output-dir result_file1.hdf5 ..."
  1>&2 echo "Example: $0 result_file_to_parquet.py test/data/ test/data/results*.hdf5"
  exit 1
fi

script="$1"
output_dir="$2"
shift 2
result_files=("$@")

if [ ! -f "$1" ]; then
  1>&2 echo "File \"$1\" does not exist!"
  exit 1
fi

tmpdir="$(mktemp -d)"

# shellcheck disable=SC2064
trap "rm -rf '$tmpdir'" EXIT

output_name="$output_dir/stripepy-call-result-tables.tar.xz"
output_dir="$tmpdir/stripepy-call-result-tables"
mkdir -p "$output_dir"


for f in "${result_files[@]}"; do
  1>&2 echo "Processing \"$f\"..."
  python3 "$script" "$f" "$output_dir"
done

1>&2 echo "Creating archive \"$output_name\"..."
tar -cf - -C "$tmpdir" "$(basename "$output_dir")" |
  xz -9 --extreme | tee "$output_name" > /dev/null

1>&2 echo 'DONE!'
