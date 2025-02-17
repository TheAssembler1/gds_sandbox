#include "config.h"

unsigned long num_files = 10;
unsigned long big_file_size_bytes = 1024;
unsigned long small_file_size_bytes = 512;

bool gen_files = NULL;
size_t file_size = 0;

char* data_movement_type_string = NULL;
// FIMXE: this should be a cmd arg
char* data_movement_op_string = "read";

unsigned int status_update_file_num = 100;

const char* data_movement_op_strings[] = {"read", "invalid_data_movement_op"};
const char* data_movement_type_strings[] = {"malloc", "gpu_direct", "mmap", "invalid_data_movement_type"};
