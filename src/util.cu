#include "util.h"

void validate_args(const char* gen_files, const char* file_size, const char* data_movement_op, const char* num_files) {
    if(strcmp(gen_files, "true") && strcmp(gen_files, "false")) {
      cpu_printf("invalid gen_files\n");
      cpu_printf("%s\n", USAGE_DETAILS);
      exit(1);
    } 
  
    if(strcmp(file_size, "small_files") && strcmp(file_size, "big_files")) {
      cpu_printf("invalid file\n");
      cpu_printf("%s\n", USAGE_DETAILS);
      exit(1);
    }
  
    bool found = false;
    for(int i = 0; i < DATA_MOVEMENT_OP_TYPE_SIZE; i++) {
      if(strcmp(data_movement_op, data_movement_op_strings[i])) {
        found = true;
        break;
      }
    }
  
    ASSERT(found, "invalid data_movement_op: %s", USAGE_DETAILS);
    ASSERT(atoi(num_files) > 0, "invalid num_files: %s", USAGE_DETAILS); 
} 


static void create_directory() {
    struct stat dir_stat;
    if(stat(DATA_OUTPUT_DIR, &dir_stat) == -1) {
      cpu_printf("creating directory %s\n", DATA_OUTPUT_DIR);
  
      if(mkdir(DATA_OUTPUT_DIR, 0777) != 0) {
        cpu_printf("failed to create directory %s", DATA_OUTPUT_DIR);
        exit(1);
      }
    } else {
      cpu_printf("directory %s already exists... skipping creation\n", DATA_OUTPUT_DIR);
    }
}
  
static void create_file(int file_num) {
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
  
    char* data = (char*)malloc(file_size * sizeof(char));
    size_t wb = fwrite(data, sizeof(char), file_size, fp);
    if(wb != file_size) {
      perror("fwrite");
      exit(1);
    }
  
    fclose(fp);
  
    free(data);
    free(file_name);
}
  
void create_data() {
    /* create main data directory */
    create_directory();
    
    for(int i = 1; i <= num_files; i++) {
      if(i % status_update_file_num == 0) {
        cpu_status_printf("created %d files\n", i);
      }
      create_file(i);
    }
}
  