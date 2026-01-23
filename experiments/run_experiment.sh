#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

input_dir="experiment_input"
start_index="${1:-0}"

if [[ "${start_index}" != "0" && "${start_index}" != "1" ]]; then
    echo "Usage: $0 [0|1]" >&2
    exit 1
fi

files=("${input_dir}"/*)
for ((i = start_index; i < ${#files[@]}; i += 2)); do
    file="${files[i]}"
    if [[ -f "${file}" ]]; then
        uv run python -m src.main "${file}"
    fi
done
