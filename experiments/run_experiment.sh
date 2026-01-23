#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

input_dir="experiment_input"

for file in "${input_dir}"/*; do
    if [[ -f "${file}" ]]; then
        uv run python -m src.main "${file}"
    fi
done
