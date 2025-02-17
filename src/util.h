#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <stdarg.h>

#define ASSERT(check, format, ...) \
    if (!(check)) { \
        printf("ASSERT FAILED: " format "\nFile: %s\nLine: %d\nFunction: %s\n", \
               ##__VA_ARGS__, __FILE__, __LINE__, __func__); \
        exit(1); \
    }

#endif