#ifndef COMMON_SSJOIN_STAGING_H_
#define COMMON_SSJOIN_STAGING_H_

#pragma once

#include <cstdint>
#include <cstddef>

void transfer_records_async(
    uint32_t **records_out,
    uint32_t **records_d_out,
    size_t &buffer_size,
    const uint32_t *input,
    size_t input_size,
    int cardinality);

#endif