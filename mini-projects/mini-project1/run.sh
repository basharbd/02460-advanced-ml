#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/src"
python -m scripts.run_partA --fast
python -m scripts.run_partB --fast
