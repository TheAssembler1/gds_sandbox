#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <cufile.h>
#include <cuda_runtime.h>

#include "config.h"
#include "read_to_gpu.h"
#include "log.h"

extern void* device_data;

void run_gpu_operations();
__global__ void simple_gpu_kernel(char* data, size_t size);

#endif