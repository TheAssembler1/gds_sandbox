#!/bin/bash

set -e
set -u

date

time ./bin/gds_sandbox false big_files

echo "Generating profile data using gprof..."
gprof ./bin/gds_sandbox gmon.out > profile_report.txt

echo "Converting profile data to dot format with gprof2dot..."
gprof2dot -f prof -o gprof_output.dot profile_report.txt

echo "Generating PNG visualization from dot file..."
dot -Tpng gprof_output.dot -o gprof_output.png

date
