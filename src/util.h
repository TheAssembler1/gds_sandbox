#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <stdarg.h>

#include "log.h"
#include "config.h"
#include "read_to_gpu.h"

#define USAGE_DETAILS \
  "Arguments:\n" \
  "  <gen_files>          : (true or false)  Set to true to generate files, false otherwise\n" \
  "  <file>               : (small_files, big_files)  Specify file type\n" \
  "  <dir>                : (single_dir, many_dir)  Specify directory type\n" \
  "  <data_movement_type> : (malloc, gpu_direct, mmap) How to move data between GPU, CPU, and storage\n" \
  "  <num_files>          : (single_dir, many_dir)  Specify number of files to operate on\n"

#define ASSERT(check, format, ...) \
    if (!(check)) { \
        printf("ASSERT FAILED: " format "\nFile: %s\nLine: %d\nFunction: %s\n", \
               ##__VA_ARGS__, __FILE__, __LINE__, __func__); \
        exit(1); \
    }

void validate_args(const char* gen_files, const char* file_size, const char* data_movement_op, const char* num_files);
void create_data();

#endif