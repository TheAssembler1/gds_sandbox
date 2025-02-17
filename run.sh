#!/bin/bash

set -e
set -u
set -x

make || { echo "Make failed, exiting..."; exit 1; }

date=

MOVEMENT_TYPE="gpu_direct"
CSV_OUTPUT_FILE="profile_output.csv"

echo "removing previous $CSV_OUTPUT_FILE"
rm -f $CSV_OUTPUT_FILE 

for MOVEMENT_TYPE in "gpu_direct" "malloc" "mmap"; do
    for ((i = 1; i <= 5; i++)); do
        echo "Running with $i files"
        ./bin/gds_sandbox false big_files "$MOVEMENT_TYPE" $i 2>> $CSV_OUTPUT_FILE
    done
done

echo "Finished all runs at $(date)"
