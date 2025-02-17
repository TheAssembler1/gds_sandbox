#!/bin/bash

set -e
set -u

make || { echo "Make failed, exiting..."; exit 1; }

date

MOVEMENT_TYPE="gpu_direct"
CSV_OUTPUT_FILE="${MOVEMENT_TYPE}_output.csv"

for i in $(seq 10 10 100); do
    echo "running with $i files"
    ./bin/gds_sandbox false big_files "$MOVEMENT_TYPE" "$i" 2> $CSV_OUTPUT_FILE
done

echo "Finished all runs at $(date)" | tee -a "$OUTPUT_FILE"


