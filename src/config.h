#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>

#define DATA_OUTPUT_DIR "./data"
#define FILE_PREFIX "/file_"
#define FILE_SUFFIX ".data"

extern unsigned long num_files;
extern unsigned long big_file_size_bytes;
extern unsigned long small_file_size_bytes;

extern bool gen_files;
extern size_t file_size;

extern char* data_movement_type_string;
extern char* data_movement_op_string;

/*
 * these have to be macros
 * they are used in the GPU functions
 */
#define data_per_thread 128
#define threads_per_block 256

extern unsigned int status_update_file_num;

typedef enum {
    READ,
    INVALID_DATA_MOVEMENT_OP,
    DATA_MOVEMENT_OP_TYPE_SIZE
} data_movement_op_t;
extern const char* data_movement_op_strings[];
  
typedef enum {
    MALLOC,
    GPU_DIRECT,
    MMAP,
    INVALID_DATA_MOVEMENT_TYPE,
    DATA_MOVEMENT_TYPE_SIZE
} data_movement_type_t;
extern const char* data_movement_type_strings[];

#endif