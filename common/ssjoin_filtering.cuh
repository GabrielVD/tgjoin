#ifndef COMMON_SSJOIN_FILTERING_H_
#define COMMON_SSJOIN_FILTERING_H_

#pragma once

#include <ssjoin_types.cuh>

__global__ void filter(
    const record_t *record_map_d,
    record_t key_start,
    record_t key_limit,
    const record_t *token_map_d,
    record_t token_limit,
    const index_record *index_d,
    float threshold,
    float overlap_factor,
    overlap_t *overlap_matrix_d,
    size_t overlap_offset,
    int* candidates_d);

#endif