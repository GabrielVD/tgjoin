#ifndef COMMON_SSJOIN_VERIFY_H_
#define COMMON_SSJOIN_VERIFY_H_

#pragma once

#include <ssjoin_types.cuh>

#define OVERLAP_PACK_SIZE (sizeof(overlap_pack) / sizeof(overlap_t))

typedef unsigned long long overlap_pack;

__global__ void verify(
    const record_t *record_map_d,
    record_pair *out_d,
    int *out_count_d,
    int *candidates_d,
    overlap_pack *overlap_pack_d,
    float threshold,
    float overlap_factor,
    size_t overlap_offset,
    size_t pack_count);

__host__ __device__ inline int verifyBlockSizeToDynamicSMemSize(int block)
{
    return 2 * sizeof(size_t) + block * (sizeof(size_t) + sizeof(record_pair) + sizeof(overlap_t));
}

#endif
