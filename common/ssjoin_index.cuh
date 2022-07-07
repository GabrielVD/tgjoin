#ifndef COMMON_SSJOIN_INDEX_H_
#define COMMON_SSJOIN_INDEX_H_

#pragma once

#include <cstdint>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

struct indexed_record
{
    uint32_t start_index;
    uint32_t remaining_tokens;
    uint32_t size;
};

__global__ void count_tokens(
    const uint32_t *records_d,
    int cardinality,
    uint32_t *count_d,
    float threshold);

inline void prefix_sum(uint32_t *first_d, uint32_t size, uint32_t *result_d)
{
    thrust::device_ptr<uint32_t> first(first_d);
    thrust::device_ptr<uint32_t> result(result_d);
    thrust::exclusive_scan(first, first + size, result);
}

#endif
