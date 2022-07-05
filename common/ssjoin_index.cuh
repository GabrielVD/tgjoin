#ifndef COMMON_SSJOIN_INDEX_H_
#define COMMON_SSJOIN_INDEX_H_

#pragma once

#include <cstdint>

__global__ void count_tokens(
    const uint32_t *records_d,
    int cardinality,
    uint32_t *count_d,
    float threshold);

#endif
