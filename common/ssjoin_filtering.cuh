#ifndef COMMON_SSJOIN_FILTERING_H_
#define COMMON_SSJOIN_FILTERING_H_

#pragma once

#include <ssjoin_types.cuh>

__global__ void filter(
    const record_t *records_d,
    record_t id_start,
    const record_t id_limit,
    const record_t *token_map_d,
    const record_t token_limit,
    const index_record *inverted_index_d,
    const float threshold,
    const float overlap_factor,
    uint8_t *matrix_tip_d,
    record_t *stats_d); // [token_probes, index_probes]

#endif