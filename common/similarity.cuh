#ifndef COMMON_SIMILARITY_CUH_
#define COMMON_SIMILARITY_CUH_

#pragma once

#include <cmath>

template<typename T>
__device__ inline T overlap_jac(float threshold, T x, T y)
{
    return ceil((threshold / (threshold + 1.0f)) * (float)(x + y));
}

#define OVERLAP(threshold, x, y) overlap_jac(threshold, x, y)

template<typename T>
__device__ inline T index_prefix_size(T set_size, float threshold)
{
    return set_size + 1 - OVERLAP(threshold, set_size, set_size);
}

#endif