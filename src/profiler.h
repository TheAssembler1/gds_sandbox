#ifndef PROFILE_H

extern long double total_gpu_func_time;
extern long double total_metadata_time;
extern long double total_data_movement_storage_to_cpu_time;
extern long double total_data_movement_cpu_to_gpu;

#define TIME_FUNC_RET(func, inc_var, ret) { \
    struct timeval start, end; \
    gettimeofday(&start, NULL); \
    ret = func; \
    gettimeofday(&end, NULL); \
    long double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 + \
                               (end.tv_usec - start.tv_usec) / 1000.0; \
    inc_var += elapsed_time; \
  }
  
  #define TIME_FUNC(func, inc_var) { \
    struct timeval start, end; \
    gettimeofday(&start, NULL); \
    func; \
    gettimeofday(&end, NULL); \
    long double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 + \
                               (end.tv_usec - start.tv_usec) / 1000.0; \
    inc_var += elapsed_time; \
  }

#define TIME_GPU_EXECUTION_FUNC(func) TIME_FUNC(func, total_gpu_func_time)
#define TIME_GPU_EXECUTION_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_gpu_func_time, ret)

#define TIME_METADATA_FUNC(func) TIME_FUNC(func, total_metadata_time)
#define TIME_METADATA_FUNC_RET(func, ret) TIME_FUNC_RET(func, total_metadata_time, ret)

#endif