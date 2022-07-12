#ifndef COMMON_SIMILARITY_CUH_
#define COMMON_SIMILARITY_CUH_

#pragma once

#include <cmath>

template<typename T>
__device__ inline T overlap_jac(T x, T y, float overlap_factor)
{
    return ceil(__fmul_ru(overlap_factor, (float)(x + y)));
}

inline float overlap_factor_jac(float threshold)
{
    return threshold / (threshold + 1.0f);
}

template<typename T>
__device__ inline T minsize_jac(T set_size, float threshold)
{
    return ceil(__fmul_ru(threshold, (float)set_size));
}

#define OVERLAP_FAC(threshold) overlap_factor_jac(threshold)
#define OVERLAP(x, y, fac) overlap_jac(x, y, fac)
#define MINSIZE(set_size, threshold) minsize_jac(set_size, threshold)

template<typename T>
__device__ inline T index_prefix_size(T set_size, float overlap_factor)
{
    return set_size + 1 - OVERLAP(set_size, set_size, overlap_factor);
}

template<typename T>
__device__ inline T prefix_size(T set_size, float threshold)
{
    return set_size + 1 - MINSIZE(set_size, threshold);
}

#endif