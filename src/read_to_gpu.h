#ifndef READ_TO_GPU
#define READ_TO_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include <errno.h>

#include "log.h"
#include "profiler.h"
#include "util.h"

void* gpu_read_direct_data(char* filepath, int file_num);
void* gpu_read_mmap_data(char* file_path, int file_num);
void* gpu_read_malloc_data(char* file_path, int file_num);

#endif