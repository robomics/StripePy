#!/usr/bin/env bash

# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

set -e
set -u
set -o pipefail

data_dir='test/data'
resolution=10000
region='chr2:120100000-122100000'

tmpdir="$(mktemp -d)"
plot_dir="$tmpdir/stripepy-plot-test-images"
mkdir "$plot_dir"

# shellcheck disable=SC2064
trap "rm -rf '$tmpdir'" EXIT

stripepy plot cm \
  "$data_dir/4DNFI9GMP2J8.mcool" \
  "$resolution" \
  "$plot_dir/contact_map.png" \
  --region "$region" \
  --force

stripepy plot cm \
  "$data_dir/4DNFI9GMP2J8.mcool" \
  "$resolution" \
  "$plot_dir/contact_map_with_seeds.png" \
  --region "$region" \
  --stripepy-hdf5 "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  --highlight-seeds \
  --force

stripepy plot cm \
  "$data_dir/4DNFI9GMP2J8.mcool" \
  "$resolution" \
  "$plot_dir/contact_map_with_stripes.png" \
  --region "$region" \
  --stripepy-hdf5 "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  --highlight-stripes \
  --force

stripepy plot cm \
  "$data_dir/4DNFI9GMP2J8.mcool" \
  "$resolution" \
  "$plot_dir/contact_map_with_stripes_no_heights.png" \
  --region "$region" \
  --stripepy-hdf5 "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  --highlight-stripes \
  --ignore-stripe-heights \
  --force

stripepy plot pd \
  "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  "$plot_dir/pseudodistribution.png" \
  --region "$region" \
  --force

stripepy plot hist \
  "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  "$plot_dir/stripe_hist.png" \
  --region "$region" \
  --force

stripepy plot hist \
  "$data_dir/results_4DNFI9GMP2J8_v3.hdf5" \
  "$plot_dir/stripe_hist_gw.png" \
  --force

tar -cf - -C "$tmpdir" "$(basename "$plot_dir")" |
  xz -9 --extreme | tee "$data_dir/stripepy-plot-test-images.tar.xz" > /dev/null
