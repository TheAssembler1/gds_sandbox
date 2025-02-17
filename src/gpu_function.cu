#include "gpu_function.h"

void* device_data = NULL;

__global__ void simple_gpu_kernel(char* data, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    gpu_printf("starting simple_gpu_kernel with index %d\n", index);
  
    if (index * data_per_thread < size) {
      size_t end_index = min((index + 1) * data_per_thread, size);
      for (size_t i = index * data_per_thread; i < end_index; i++) {
        gpu_printf("%c", data[i]);
      }
    }
}


static void exec_gpu_function() {
  cudaError_t cuda_status;

  /* run gpu kernel */
  size_t block_size = threads_per_block;  
  size_t grid_size = (file_size + data_per_thread * block_size - 1) / (data_per_thread * block_size);
  cpu_printf("file_size: %d, threads_per_block: %d block_size: %zu, grid_size: %zu\n", 
              file_size, threads_per_block, block_size, grid_size);
  simple_gpu_kernel<<<grid_size, block_size>>>((char*)device_data, file_size);

  cuda_status = cudaGetLastError();
  ASSERT(cuda_status == cudaSuccess, "failed to launch kernel status: %d", cuda_status);

  cuda_status = cudaDeviceSynchronize();
  ASSERT(cuda_status == cudaSuccess, "failed to cudaDeviceSynchronize error_string: %s", cudaGetErrorString(cuda_status));
}

void run_gpu_operations() {
    CUfileError_t file_status;
    data_movement_op_t data_movement_op = INVALID_DATA_MOVEMENT_OP;
    data_movement_type_t data_movement_type = INVALID_DATA_MOVEMENT_TYPE;
  
    /* initialize  state of critical performance path */
    file_status = cuFileDriverOpen();
    ASSERT(file_status.err == CU_FILE_SUCCESS, "failed to cuFileDriverOpen status: %d", file_status);
    
    if(!strcmp(data_movement_type_string, "malloc")) {
      data_movement_type = MALLOC;
    } else if(!strcmp(data_movement_type_string, "gpu_direct")) {
      data_movement_type = GPU_DIRECT;
    } else if(!strcmp(data_movement_type_string, "mmap")) {
      data_movement_type = MMAP;
    } else {
      ASSERT(false, "invalid data movement type %s", data_movement_type_string);
    }
  
    if(!strcmp(data_movement_op_string, "read")) {
      data_movement_op = READ;
    } else {
      ASSERT(false, "invalid data movement operation %s", data_movement_op_string);
    }
  
    for(int i = 1; i <= num_files; i++) {
      if(i % status_update_file_num == 0) {
        cpu_status_printf("processed %d files\n", i);
      }
  
      /* get file name */
      char file_num_str[32];
      sprintf(file_num_str, "%d", i);
      size_t file_name_len = strlen(DATA_OUTPUT_DIR) + strlen(FILE_PREFIX) + strlen(file_num_str) + strlen(FILE_SUFFIX)  + 1;
      char* file_name = (char*)malloc(file_name_len * sizeof(char));
    
      strcpy(file_name, DATA_OUTPUT_DIR);
      strcat(file_name, FILE_PREFIX);
      strcat(file_name, file_num_str);
      strcat(file_name, FILE_SUFFIX);
    
      cpu_printf("reading file %s\n", file_name);
  
      /* select data movement operation */
      switch(data_movement_op) {
        case READ:
          switch(data_movement_type) {
            case MALLOC:
              device_data = gpu_read_malloc_data(file_name, i);
              break;
            case GPU_DIRECT:
              device_data = gpu_read_direct_data(file_name, i);
              break;
            case MMAP:
              device_data = gpu_read_mmap_data(file_name, i);
              break;
            case INVALID_DATA_MOVEMENT_TYPE:
              exit(1);
              break;
            default:
              exit(1);
              break;
          }
        break;
        case INVALID_DATA_MOVEMENT_OP:
          exit(1);
          break;
        default:
          exit(1);
          break;
      }
  
      /* execute the gpu function */
      TIME_GPU_EXECUTION_FUNC(exec_gpu_function());
  
      /* free memory buffer on GPU device */
      cudaFree(device_data);
    }
  
    cuFileDriverClose();
}