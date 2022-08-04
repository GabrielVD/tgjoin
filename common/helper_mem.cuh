#ifndef COMMON_HELPER_MEM_H_
#define COMMON_HELPER_MEM_H_

#pragma once

#include <stdlib.h>
#include <ssjoin_types.cuh>
#include <cmath>

#define CACHE_LINE 128
#define BYTES_R(n) ((n) * sizeof(record_t))
#define BYTES_I(n) ((n) * sizeof(index_record))
#define BYTES_O(n) ((n) * sizeof(overlap_t))

typedef unsigned char byte_t;

template<typename T, uintptr_t N>
inline T *aligned_up(T *ptr)
{
    return (T*)(((uintptr_t)(ptr) + (N - 1)) & -N);
}

inline size_t byte_diff(void *high, void *low)
{
    return (byte_t*)high - (byte_t*)low;
}

__host__ __device__ inline size_t tri_rowstart(size_t i)
{
    return (i * (i - 1)) / 2;
}

inline size_t tri_maxfit(size_t capacity)
{
    return std::floor(std::sqrt(2 * capacity + .25) + .5);
}

__device__ inline size_t tri_maxfit_d(const size_t capacity)
{
    double capacity_lf;
    {
        unsigned long long capacity_ull{capacity};
        capacity_lf = __ull2double_ru(capacity_ull);
    }
    
    return
        __double2ull_rd(
            __dadd_ru(
                __dsqrt_ru(
                    __fma_ru(2.0, capacity_lf, .25)),
                .5));
}

#endif
