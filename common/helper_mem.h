#ifndef COMMON_HELPER_MEM_H_
#define COMMON_HELPER_MEM_H_

#pragma once

#include <stdlib.h>
#include <ssjoin_index.cuh>

inline void safe_free(void **ptr, size_t *size)
{
    free(*ptr);
    *ptr = NULL;
    *size = 0;
}

#define BYTES_U(n) ((n) * sizeof(uint32_t))
#define BYTES_INDEX(n) ((n) * sizeof(index_record))
#define SAFE_FREE(ptr, size) safe_free((void **)ptr, size)
#define MALLOC_U(size) ((uint32_t *)malloc(BYTES_U(size)))

#endif
