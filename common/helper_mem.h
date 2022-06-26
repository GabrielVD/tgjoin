#ifndef COMMON_HELPER_MEM_H_
#define COMMON_HELPER_MEM_H_

#include <stdlib.h>

inline void safe_free(void **ptr, size_t *size)
{
    free(*ptr);
    *ptr = NULL;
    *size = 0;
}

#define SAFE_FREE(ptr, size) safe_free((void **)ptr, size)

#endif
