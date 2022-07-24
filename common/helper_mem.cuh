#ifndef COMMON_HELPER_MEM_H_
#define COMMON_HELPER_MEM_H_

#pragma once

#include <stdlib.h>
#include <ssjoin_types.cuh>
#include <cmath>

#define BYTES_R(n) ((n) * sizeof(record_t))
#define BYTES_O(n) ((n) * sizeof(overlap_t))
#define BYTES_I(n) ((n) * sizeof(index_record))

__host__ __device__ inline size_t tri_rowstart(size_t i)
{
    return (i * (i - 1)) / 2;
}

inline size_t tri_maxfit(size_t capacity)
{
    return std::floor(std::sqrt(2 * capacity + .25) + .5);
}

#endif
