#ifndef LOG_H
#define LOG_H

#define PROFILE_PREFIX "PROFILE INFO: "

// NOTE: printing macros
#undef DEBUG_KERNEL_FUNC
#define CPU_DEBUG
#undef CPU_STATUS_DEBUG

#ifdef CPU_DEBUG
    #define cpu_printf(f_, ...) printf((f_), ##__VA_ARGS__)
#else
    #define cpu_printf(f_, ...) (void)0
#endif

#ifdef CPU_STATUS_DEBUG
    #define cpu_status_printf(f_, ...) printf((f_), ##__VA_ARGS__)
#else
    #define cpu_status_printf(f_, ...) (void)0
#endif

#ifdef GPU_DEBUG
    #define gpu_printf(f_, ...) printf((f_), ##__VA_ARGS__)
#else
    #define gpu_printf(f_, ...) /* Do nothing */
#endif

#define csv_printf(column_type, f_, ...) \
    do { \
        fprintf(stderr, "%s,", column_type); \
        fprintf(stderr, f_, ##__VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } while (0)

#endif