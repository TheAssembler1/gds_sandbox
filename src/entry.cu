#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cufile.h>
#include <cuda_runtime.h>

#define USAGE_DETAILS \
  "Arguments:\n" \
  "  <gen_files>  : (true or false)  Set to true to generate files, false otherwise\n" \
  "  <file>       : (small_files, big_files)  Specify file type\n" \
  "  <dir>        : (single_dir, many_dir)  Specify directory type"

#define DATA_OUTPUT_DIR "./data"
#define FILE_PREFIX "/file_"
#define FILE_SUFFIX ".data"

// FIXME: this should be a cmd arg
unsigned int num_files = 100;
unsigned int small_file_size_bytes = 512;
unsigned int big_file_size_bytes = 4096;

void validate_args(const char* gen_files, const char* file);
void create_data(const char* file);

__global__ void simple_gpu_kernel() {
  printf("<==============================>\n");
  printf("starting simple_gpu_kernel\n");

  printf("finishing simple_gpu_kernel\n");
  printf("<==============================>\n");
}

int main(int argc, char** argv) {
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

  /* run gpu kernel */
  simple_gpu_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}

static void create_directory(const char* dir) {
  struct stat dir_stat;
  if(stat(dir, &dir_stat) == -1) {
    printf("creating directory %s\n", dir);

    if(mkdir(dir, 0777) != 0) {
      printf("failed to create directory %s", dir);
      exit(1);
    }
  } else {
    printf("directory %s already exists... skipping creation\n", dir);
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

  printf("creating file %s\n", file_name);

  FILE* fp = fopen(file_name, "wb");
  if(fp == NULL) {
    printf("failed to open file %s\n", file_name);
    exit(1);
  }
  
  int data_size_bytes = small_file_size_bytes;
  if(!strcmp(file, "big_files")) {
    data_size_bytes = big_file_size_bytes;
  }

  char* data = (char*)malloc(data_size_bytes * sizeof(char));
  size_t wb = fwrite(data, sizeof(char), data_size_bytes, fp);
  if(wb != data_size_bytes) {
    printf("failed to write data file %s\n", file_name);
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
    printf("invalid gen_files\n");
    printf("%s\n", USAGE_DETAILS);
    exit(1);
  } 

  if(strcmp(file, "small_files") && strcmp(file, "big_files")) {
    printf("invalid file\n");
    printf("%s\n", USAGE_DETAILS);
    exit(1);
  }
}
