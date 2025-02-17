#!/bin/bash

set -e
set -u
set -x

START_NUM_FILES=1000
END_NUM_FILES=10000
INC_NUM_FILES=1000

make || { echo "Make failed, exiting..."; exit 1; }

MOVEMENT_TYPE="gpu_direct"
CSV_OUTPUT_FILE="profile_output.csv"

echo "removing previous $CSV_OUTPUT_FILE"
rm -f $CSV_OUTPUT_FILE 

#./bin/gds_sandbox true big_files "$MOVEMENT_TYPE" $END_NUM_FILES 2>> $CSV_OUTPUT_FILE

for MOVEMENT_TYPE in "gpu_direct" "malloc" "mmap"; do
    for ((i = $START_NUM_FILES; i <= $END_NUM_FILES; i += $INC_NUM_FILES)); do
        echo "Running with $i files"
        ./bin/gds_sandbox false big_files "$MOVEMENT_TYPE" $i 2>> $CSV_OUTPUT_FILE
    done
done

echo "Finished all runs at $(date)"
