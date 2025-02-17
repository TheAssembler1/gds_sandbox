#!/bin/bash

set -e
set -u

make || { echo "Make failed, exiting..."; exit 1; }

date

MOVEMENT_TYPE="mmap"
CSV_OUTPUT_FILE="profile_output.csv"

./bin/gds_sandbox false big_files "$MOVEMENT_TYPE" 10 2> $CSV_OUTPUT_FILE

echo "Finished all runs at $(date)"
