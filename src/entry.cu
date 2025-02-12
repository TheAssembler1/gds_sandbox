#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <cufile.h>
#include <cuda_runtime.h>

/*
 * Best Practice: https://docs.nvidia.com/gpudirect-storage/best-practices-guide
 * API Reference: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
 * API Reference PDF: https://docs.nvidia.com/cuda/archive/11.6.0/pdf/cuFile_API.pdf
 */

#define USAGE_DETAILS \
  "Arguments:\n" \
  "  <gen_files>  : (true or false)  Set to true to generate files, false otherwise\n" \
  "  <file>       : (small_files, big_files)  Specify file type\n" \
  "  <dir>        : (single_dir, many_dir)  Specify directory type"

#define DATA_OUTPUT_DIR "./data"
#define FILE_PREFIX "/file_"
#define FILE_SUFFIX ".data"

#define PROFILE_PREFIX "PROFILE INFO: "

// NOTE: printing macros
#undef DEBUG_KERNEL_FUNC
#undef CPU_DEBUG

#ifdef CPU_DEBUG
    #define cpu_printf(fmt, ...) printf(fmt, __VA_ARGS__)
#else
    #define cpu_printf(fmt, ...) /* Do nothing */
#endif

#ifdef GPU_DEBUG
    #define gpu_printf(fmt, ...) printf(fmt, __VA_ARGS__)
#else
    #define gpu_printf(fmt, ...) /* Do nothing */
#endif


// FIXME: this should be a cmd arg
unsigned int num_files = 10000;
unsigned int small_file_size_bytes = 512;
unsigned int big_file_size_bytes = 4096;

void validate_args(const char* gen_files, const char* file);
void create_data(const char* file);
void run_gpu_operations();

// FIXME: this should be a cmd arg
#define DATA_PER_THREAD 128
#define THREADS_PER_BLOCK 256

// FIXME: this should be a parameter
#define DATA_MOVEMENT_TYPE "posix"
#define DATA_MOVEMENT_OP "read"

long double total_data_movement_time = 0;

#define TIME_FUNC(func, inc_var) { \
  struct timeval start, end; \
  gettimeofday(&start, NULL); \
  func; \
  gettimeofday(&end, NULL); \
  long double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 + \
                             (end.tv_usec - start.tv_usec) / 1000.0; \
  inc_var += elapsed_time; \
}

#define TIME_DATA_MOVEMENT_FUNC(func) TIME_FUNC(func, total_data_movement_time)

__global__ void simple_gpu_kernel(char* data, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  gpu_printf("starting simple_gpu_kernel with index %d\n", index);

  if (index * DATA_PER_THREAD < size) {
    size_t end_index = min((index + 1) * DATA_PER_THREAD, size);
    for (size_t i = index * DATA_PER_THREAD; i < end_index; i++) {
      gpu_printf("%c", data[i]);
    }
  }
}

static void run(int argc, char** argv) {
  if(argc < 3) {
    printf("%s\n", USAGE_DETAILS);
    exit(1);
  }
  
  /* grab cmd args */
  const char* gen_files = argv[1];
  const char* file = argv[2];

  validate_args(gen_files, file);

  /* check and generate files and folders */
  if(!strcmp(gen_files, "true")) {
    printf("generating test data\n");
    create_data(file);
  } else {
    printf("assuming test data is already generated\n");
  }

  /* config info should not be hidden behind debug */
  // FIXME: assumes big files
  printf("total files: %d\n", num_files);
  printf("size of each file: %d bytes\n", big_file_size_bytes);
  printf("total data movement size: %d bytes\n", num_files * big_file_size_bytes);
  printf("data movement operation: %s, data movement type\n", DATA_MOVEMENT_OP, DATA_MOVEMENT_TYPE);

  run_gpu_operations();
}

int main(int argc, char** argv) {
  long double total_runtime = 0;
  TIME_FUNC(run(argc, argv), total_runtime);

  /* profiling info should not be hidden behind debug */
  printf("%s total runtime: %Lf ms\n", PROFILE_PREFIX, total_runtime);

  return 0;
}

typedef enum {
  READ,
  INVALID_DATA_MOVEMENT_OP
} data_movement_op_t;

typedef enum {
  MALLOC,
  INVALID_DATA_MOVEMENT_TYPE
} data_movement_type_t;

// FIXME: file size
size_t file_size = 0;
void *device_data = NULL;

static void gpu_read_malloc_data(char* file_path) {
  int fd;
  void *host_data = NULL;

  fd = open(file_path, O_RDONLY);
  if (fd == -1) {
      perror("error opening file");
      exit(1);
  }

  file_size = lseek(fd, 0, SEEK_END);
  if (file_size == -1) {
      perror("error getting file size");
      close(fd);
      exit(1);
  }
  lseek(fd, 0, SEEK_SET);

  host_data = malloc(file_size);
  if (host_data == NULL) {
      perror("error allocating memory for file data");
      close(fd);
      exit(1);
  }

  ssize_t bytes_read = read(fd, host_data, file_size);
  if (bytes_read != file_size) {
      perror("Error reading file");
      free(host_data);
      close(fd);
      exit(1);
  }

  cudaError_t cuda_status = cudaMalloc(&device_data, file_size);
  if (cuda_status != cudaSuccess) {
    cpu_printf("failed CUDA malloc: %s\n", cudaGetErrorString(cuda_status));
      free(host_data);
      close(fd);
      exit(1);
  }

  cuda_status = cudaMemcpy(device_data, host_data, file_size, cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    cpu_printf("CUDA memcpy failed: %s\n", cudaGetErrorString(cuda_status));
      cudaFree(device_data);
      free(host_data);
      close(fd);
      exit(1);
  }

  free(host_data);
  close(fd);
}

void run_gpu_operations() {
  CUfileError_t file_status;
  cudaError_t cuda_status;
  data_movement_op_t data_movement_op = INVALID_DATA_MOVEMENT_OP;
  data_movement_type_t data_movement_type = INVALID_DATA_MOVEMENT_TYPE;

  /* initialize  state of critical performance path */
  file_status = cuFileDriverOpen();
  if(file_status.err != CU_FILE_SUCCESS) {
    cpu_printf("failed to initialize cuFileDriver\n");
    exit(1);
  } else {
    cpu_printf("successfully initialize cuFileDriver\n");
  }

  cpu_printf("data movement type: %s, operation: %s\n", DATA_MOVEMENT_TYPE, DATA_MOVEMENT_OP);

  if(!strcmp(DATA_MOVEMENT_TYPE, "posix")) {
    data_movement_type = MALLOC;
  } else {
    cpu_printf("invalid data movement type\n");
    exit(1);
  }

  if(!strcmp(DATA_MOVEMENT_OP, "read")) {
    data_movement_op = READ;
  } else {
    cpu_printf("invalid data movement operation\n");
    exit(1);
  }

  for(int i = 1; i <= num_files; i++) {
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
      case MALLOC:
        switch(data_movement_type) {
          case READ:
            gpu_read_malloc_data(file_name);
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

    /* run gpu kernel */
    size_t block_size = THREADS_PER_BLOCK;  
    size_t grid_size = (file_size + DATA_PER_THREAD * block_size - 1) / (DATA_PER_THREAD * block_size);
    cpu_printf("block size: %d, grid_size: %d\n", block_size, grid_size);
    simple_gpu_kernel<<<grid_size, block_size>>>((char*)device_data, file_size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      cpu_printf("failed CUDA kernel launch: %s\n", cudaGetErrorString(cuda_status));
    } else {
      cpu_printf("successfully launched kernel.\n");
    }

    cuda_status = cudaDeviceSynchronize();
    if(cuda_status != cudaSuccess) {
      const char* error_string = cudaGetErrorString(cuda_status);
      cpu_printf("error found when syncing device: %s\n", error_string);
    } else {
      cpu_printf("no errors when syncing device\n");
    }

    /* free memory buffer on GPU device */
    cudaFree(device_data);
  }

  cuFileDriverClose();
}

static void create_directory(const char* dir) {
  struct stat dir_stat;
  if(stat(dir, &dir_stat) == -1) {
    cpu_printf("creating directory %s\n", dir);

    if(mkdir(dir, 0777) != 0) {
      cpu_printf("failed to create directory %s", dir);
      exit(1);
    }
  } else {
    cpu_printf("directory %s already exists... skipping creation\n", dir);
  }
}

static void create_file(const char* file, int file_num) {
  char file_num_str[32];
  sprintf(file_num_str, "%d", file_num);
  size_t file_name_len = strlen(DATA_OUTPUT_DIR) + strlen(FILE_PREFIX) + strlen(file_num_str) + strlen(FILE_SUFFIX)  + 1;
  char* file_name = (char*)malloc(file_name_len * sizeof(char));

  strcpy(file_name, DATA_OUTPUT_DIR);
  strcat(file_name, FILE_PREFIX);
  strcat(file_name, file_num_str);
  strcat(file_name, FILE_SUFFIX);

  cpu_printf("creating file %s\n", file_name);

  FILE* fp = fopen(file_name, "wb");
  if(fp == NULL) {
    cpu_printf("failed to open file %s\n", file_name);
    exit(1);
  }
  
  int data_size_bytes = small_file_size_bytes;
  if(!strcmp(file, "big_files")) {
    data_size_bytes = big_file_size_bytes;
  }

  char* data = (char*)malloc(data_size_bytes * sizeof(char));
  size_t wb = fwrite(data, sizeof(char), data_size_bytes, fp);
  if(wb != data_size_bytes) {
    cpu_printf("failed to write data file %s\n", file_name);
    exit(1);
  }

  fclose(fp);

  free(data);
  free(file_name);
}

void create_data(const char* file) {
  /* create main data directory */
  create_directory(DATA_OUTPUT_DIR);
  
  for(int i = 1; i <= num_files; i++) {
    create_file(file, i);
  }
}

void validate_args(const char* gen_files, const char* file) {
  if(strcmp(gen_files, "true") && strcmp(gen_files, "false")) {
    cpu_printf("invalid gen_files\n");
    cpu_printf("%s\n", USAGE_DETAILS);
    exit(1);
  } 

  if(strcmp(file, "small_files") && strcmp(file, "big_files")) {
    cpu_printf("invalid file\n");
    cpu_printf("%s\n", USAGE_DETAILS);
    exit(1);
  }
}
