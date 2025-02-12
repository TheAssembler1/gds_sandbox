#!/bin/bash

set -e
set -u

make || { echo "Make failed, exiting..."; exit 1; }

date

MOVEMENT_TYPE="posix"
OUTPUT_FILE="${MOVEMENT_TYPE}_output.txt"

for i in $(seq 1 10); do
    echo "running with $i files"
    ./bin/gds_sandbox false big_files "$MOVEMENT_TYPE" "$i" >> output.txt
done

echo "Finished all runs at $(date)" | tee -a "$OUTPUT_FILE"
