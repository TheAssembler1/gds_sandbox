#ifndef PROFILE_H

extern long double total_gpu_func_time;
extern long double total_metadata_time;
extern long double total_data_movement_storage_to_cpu_time;
extern long double total_data_movement_storage_to_gpu_time;
extern long double total_data_movement_cpu_to_gpu;
extern long double total_gpu_malloc_time;
extern long double total_cpu_malloc_time;

#define TIME_FUNC_RET(func, inc_var, ret) do { \
  struct timeval start, end; \
  gettimeofday(&start, NULL); \
  ret = (func); /* Ensure func is evaluated correctly */ \
  gettimeofday(&end, NULL); \
  long double elapsed_time = ((long double)(end.tv_sec - start.tv_sec) * 1000.0) + \
                             ((long double)(end.tv_usec - start.tv_usec) / 1000.0); \
  (inc_var) += elapsed_time; \
} while (0)

  
#define TIME_FUNC(func, inc_var) do { \
  struct timeval start, end; \
  gettimeofday(&start, NULL); \
  (func); /* Ensure func is treated as an expression */ \
  gettimeofday(&end, NULL); \
  long double elapsed_time = ((long double)(end.tv_sec - start.tv_sec) * 1000.0) + \
                             ((long double)(end.tv_usec - start.tv_usec) / 1000.0); \
  (inc_var) += elapsed_time; \
} while (0)


#define TIME_GPU_EXECUTION_FUNC(func) TIME_FUNC(func, total_gpu_func_time)
#define TIME_GPU_EXECUTION_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_gpu_func_time, ret)

#define TIME_METADATA_FUNC(func) TIME_FUNC(func, total_metadata_time)
#define TIME_METADATA_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_metadata_time, ret)

#define TIME_GPU_MALLOC_FUNC(func) TIME_FUNC(func, total_gpu_malloc_time)
#define TIME_GPU_MALLOC_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_gpu_malloc_time, ret)

#define TIME_CPU_MALLOC_FUNC(func) TIME_FUNC(func, total_cpu_malloc_time)
#define TIME_CPU_MALLOC_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_cpu_malloc_time, ret)

#define TIME_DATA_MOVEMENT_STORAGE_TO_CPU_FUNC(func) TIME_FUNC(func, total_data_movement_storage_to_cpu_time)
#define TIME_DATA_MOVEMENT_STORAGE_TO_CPU_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_data_movement_storage_to_cpu_time, ret)

#define TIME_DATA_MOVEMENT_STORAGE_TO_GPU_FUNC(func) TIME_FUNC(func, total_data_movement_storage_to_gpu_time)
#define TIME_DATA_MOVEMENT_STORAGE_TO_GPU_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_data_movement_storage_to_gpu_time, ret)

#define TIME_DATA_MOVEMENT_CPU_TO_GPU_FUNC(func) TIME_FUNC(func, total_data_movement_cpu_to_gpu)
#define TIME_DATA_MOVEMENT_CPU_TO_GPU_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_data_movement_cpu_to_gpu, ret)

#endif