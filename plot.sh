#!/bin/bash

set -e

echo "parsing input data"
python3 plot.py
echo "creating png plots"
gnuplot plot_script.gnu