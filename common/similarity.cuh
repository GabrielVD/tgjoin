#ifndef COMMON_SIMILARITY_CUH_
#define COMMON_SIMILARITY_CUH_

#pragma once

#include <cmath>

template<typename T>
__device__ inline T overlap_jac(T x, T y, float threshold)
{
    return ceil((threshold / (threshold + 1.0f)) * (float)(x + y));
}

template<typename T>
__device__ inline T minsize_jac(T set_size, float threshold)
{
    return ceil(threshold * (float)set_size);
}

#define OVERLAP(x, y, threshold) overlap_jac(x, y, threshold)
#define MINSIZE(set_size, threshold) minsize_jac(set_size, threshold)

template<typename T>
__device__ inline T index_prefix_size(T set_size, float threshold)
{
    return set_size + 1 - OVERLAP(set_size, set_size, threshold);
}

template<typename T>
__device__ inline T prefix_size(T set_size, float threshold)
{
    return set_size + 1 - MINSIZE(set_size, threshold);
}

#endif