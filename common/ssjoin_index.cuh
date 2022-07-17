#ifndef COMMON_SSJOIN_INDEX_H_
#define COMMON_SSJOIN_INDEX_H_

#pragma once

#include <ssjoin_types.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

struct index_record
{
    record_t id;
    record_t size;
    record_t remaining_tokens;
};

__global__ void count_tokens(
    const record_t *records_d,
    int cardinality,
    record_t *count_d,
    float overlap_factor);

__global__ void make_index(
    const record_t *records_d,
    const int cardinality,
    const record_t *token_map_d,
    const float overlap_factor,
    record_t *count_d,
    index_record *inverted_index_d);

inline void prefix_sum(record_t *first_d, record_t size, record_t *result_d)
{
    thrust::device_ptr<record_t> first(first_d);
    thrust::device_ptr<record_t> result(result_d);
    thrust::inclusive_scan(first, first + size, result);
}

#endif
