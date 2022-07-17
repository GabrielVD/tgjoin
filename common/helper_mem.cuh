#ifndef COMMON_HELPER_MEM_H_
#define COMMON_HELPER_MEM_H_

#pragma once

#include <stdlib.h>
#include <ssjoin_types.h>
#include <cmath>

#define BYTES_U(n) ((n) * sizeof(record_t))
#define BYTES_INDEX(n) ((n) * sizeof(index_record))
#define SAFE_FREE(ptr, size) safe_free((void **)ptr, size)
#define MALLOC_U(size) ((record_t *)malloc(BYTES_U(size)))

inline void safe_free(void **ptr, size_t *size)
{
    free(*ptr);
    *ptr = NULL;
    *size = 0;
}

__host__ __device__ inline size_t tri_rowstart(size_t i)
{
    return (i * (i - 1)) / 2;
}

inline size_t tri_maxfit(size_t capacity)
{
    return std::floor(std::sqrt(2 * capacity + .25) + .5);
}

#endif
