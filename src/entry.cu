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

#include "log.h"
#include "profiler.h"
#include "read_to_gpu.h"
#include "gpu_function.h"
#include "util.h"

/*
 * Best Practice: https://docs.nvidia.com/gpudirect-storage/best-practices-guide
 * API Reference: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
 * API Reference PDF: https://docs.nvidia.com/cuda/archive/11.6.0/pdf/cuFile_API.pdf
 */

static int run(int argc, char** argv) {
  if(argc < 5) {
    printf("%s\n", USAGE_DETAILS);
    exit(1);
  }

  validate_args(argv[1], argv[2], argv[3], argv[4]);
  
  /* grab cmd args */
  if(!strcmp(argv[1], "true")) {
    gen_files = true;
  } else {
    gen_files = false;
  }

  if(!strcmp(argv[2], "big_files")) {
    file_size = big_file_size_bytes;
  } else {
    file_size = small_file_size_bytes;
  }

  data_movement_type_string = argv[3];
  num_files = atoi(argv[4]);

  /* check and generate files and folders */
  if(gen_files) {
    printf("generating test data with %lu files each files is %lu bytes\n", num_files, file_size);
    create_data();
  } else {
    printf("assuming test data is already generated\n");
  }

  /* config info should not be hidden behind debug */
  printf("total files: %lu\n", num_files);
  printf("size of each file: %lu bytes\n", file_size);
  printf("total data movement size: %lu bytes %.2f megabytes\n", 
    num_files * file_size, 
    (num_files * file_size) / (1024.0 * 1024));
  printf("data movement operation: %s, data movement type %s\n", data_movement_op_string, data_movement_type_string);
  printf("CSV LAYOUT: %s\n", CSV_LAYOUT);

  run_gpu_operations();

  return 0;
}

int main(int argc, char** argv) {
  long double total_runtime = 0;
  int ret;
  TIME_FUNC_RET(run(argc, argv), total_runtime, ret);

  printf("logging profiling info to csv\n");

  csv_printf("%s,%Lf", "total_runtime", total_runtime);
  csv_printf("%s,%Lf", "total_gpu_func_time", total_gpu_func_time);
  csv_printf("%s,%Lf", "total_metadata_time", total_metadata_time);
  csv_printf("%s,%Lf", "total_data_movement_storage_to_cpu_time", total_data_movement_storage_to_cpu_time);
  csv_printf("%s,%Lf", "total_data_movement_storage_to_gpu_time", total_data_movement_storage_to_gpu_time);
  csv_printf("%s,%Lf", "total_data_movement_cpu_to_gpu", total_data_movement_cpu_to_gpu);
  csv_printf("%s,%Lf", "total_cpu_malloc_time", total_cpu_malloc_time);
  csv_printf("%s,%Lf", "total_gpu_malloc_time", total_gpu_malloc_time);

  return ret;
}