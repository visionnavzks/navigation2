#!/usr/bin/env bash

set -euo pipefail

rust_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
c_root="$(cd "$rust_root/../libccd" && pwd)"

gjk_cycles="${1:-10}"
mpr_cycles="${2:-10000}"

if [[ ! -x "$c_root/build/src/testsuites/bench" || ! -x "$c_root/build/src/testsuites/bench2" ]]; then
  cmake -S "$c_root" -B "$c_root/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DENABLE_DOUBLE_PRECISION=OFF >/dev/null
  cmake --build "$c_root/build" --target bench bench2 -j"$(nproc)" >/dev/null
fi

rust_gjk_ns="$(cd "$rust_root" && cargo run --release --example bench_compare -- gjk "$gjk_cycles" | awk -F= '/total_ns=/{print $2}')"
rust_mpr_ns="$(cd "$rust_root" && cargo run --release --example bench_compare -- mpr "$mpr_cycles" | awk -F= '/total_ns=/{print $2}')"

c_gjk_ns="$(cd "$c_root" && ./build/src/testsuites/bench "$gjk_cycles" | awk 'BEGIN{sum=0} /^[0-9][0-9]:/ {sum += $2 * 1000000000 + $3} END {printf("%.0f", sum)}')"
c_mpr_ns="$(cd "$c_root" && ./build/src/testsuites/bench2 "$mpr_cycles" | awk 'BEGIN{sum=0} /^[0-9][0-9]:/ {sum += $2 * 1000000000 + $3} END {printf("%.0f", sum)}')"

cat <<EOF
algorithm,implementation,cycles,total_ns
gjk_penetration,c,$gjk_cycles,$c_gjk_ns
gjk_penetration,rust,$gjk_cycles,$rust_gjk_ns
mpr_penetration,c,$mpr_cycles,$c_mpr_ns
mpr_penetration,rust,$mpr_cycles,$rust_mpr_ns
EOF