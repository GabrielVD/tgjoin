#ifndef COMMON_SSJOIN_VERIFY_H_
#define COMMON_SSJOIN_VERIFY_H_

#pragma once

#include <ssjoin_types.cuh>

#define VERIFY_PACK_SIZE (sizeof(unsigned long long) / sizeof(overlap_t))

__global__ void verify(
    overlap_t *overlap_matrix_d,
    size_t overlap_count);

__host__ __device__ inline int verifyBlockSizeToDynamicSMemSize(int block)
{
    return (block * VERIFY_PACK_SIZE + block - 1) * (sizeof(size_t) + sizeof(overlap_t))
        + sizeof(size_t);
}

#endif
