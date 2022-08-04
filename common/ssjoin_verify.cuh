#ifndef COMMON_SSJOIN_VERIFY_H_
#define COMMON_SSJOIN_VERIFY_H_

#pragma once

#include <ssjoin_types.cuh>

#define OVERLAP_PACK_SIZE (sizeof(overlap_pack) / sizeof(overlap_t))

typedef unsigned long long overlap_pack;

__global__ void verify(
    int *buffer_d,
    overlap_pack *overlap_pack_d,
    size_t pack_count);

__host__ __device__ inline int verifyBlockSizeToDynamicSMemSize(int block)
{
    return 2 * sizeof(int) + block * (sizeof(size_t) + sizeof(record_pair));
}

#endif
