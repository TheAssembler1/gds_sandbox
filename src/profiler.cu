#include "profiler.h"

long double total_gpu_func_time = 0;
long double total_metadata_time = 0;
long double total_data_movement_storage_to_cpu_time = 0;
long double total_data_movement_storage_to_gpu_time = 0;
long double total_data_movement_cpu_to_gpu = 0;
long double total_gpu_malloc_time = 0;
long double total_cpu_malloc_time = 0;