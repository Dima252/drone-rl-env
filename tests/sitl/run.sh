#!/usr/bin/env bash
# Fly a policy on ArduCopter SITL and real MAVROS. See README.md in this folder.
#
#   tests/sitl/run.sh <export-dir>           Run the full suite.
#   tests/sitl/run.sh <export-dir> --bench   Measure the control loop only.
set -euo pipefail

if [[ $# -lt 1 || ! -f "$1/model.onnx" || ! -f "$1/policy_meta.npz" ]]; then
  echo "Usage: $0 <export-dir> [--bench]" >&2
  echo "<export-dir> must hold model.onnx and policy_meta.npz." >&2
  exit 2
fi

# Docker Desktop on Windows needs native paths, and Git Bash must not rewrite them.
export MSYS_NO_PATHCONV=1
native_path() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }

sitl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(native_path "$sitl_dir/../../drone_rl_workspace")"
policy="$(native_path "$1")"
mkdir -p "$sitl_dir/out"

script=sitl_suite.py
if [[ "${2:-}" == "--bench" ]]; then
  script=loop_bench.py
fi

docker build -t drone-sitl "$(native_path "$sitl_dir")"
docker run --rm \
  -e POLICY_SECONDS="${POLICY_SECONDS:-10}" \
  -e WAYPOINT_SECONDS="${WAYPOINT_SECONDS:-20}" \
  -v "$workspace:/ws:ro" \
  -v "$(native_path "$sitl_dir"):/test:ro" \
  -v "$policy:/policy:ro" \
  -v "$(native_path "$sitl_dir/out"):/out" \
  drone-sitl bash -c "source /opt/ros/humble/setup.bash && python3 -u /test/$script"
