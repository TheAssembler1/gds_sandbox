#!/bin/bash

set -e
set -u
set -x

echo "removing previous png plots"
rm -f *.png

python3 plot.py