#ifndef COMMON_SSJOIN_INDEX_H_
#define COMMON_SSJOIN_INDEX_H_

#pragma once

#include <cstdint>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

struct index_record
{
    uint32_t id;
    uint32_t size;
    uint32_t remaining_tokens;
};

__global__ void count_tokens(
    const uint32_t *records_d,
    int cardinality,
    uint32_t *count_d,
    float threshold);

__global__ void make_index(
    const uint32_t *records_d,
    const int cardinality,
    const uint32_t *token_map_d,
    const float threshold,
    uint32_t *count_d,
    index_record *inverted_index_d);

inline void prefix_sum(uint32_t *first_d, uint32_t size, uint32_t *result_d)
{
    thrust::device_ptr<uint32_t> first(first_d);
    thrust::device_ptr<uint32_t> result(result_d);
    thrust::inclusive_scan(first, first + size, result);
}

#endif
