#ifndef COMMON_SSJOIN_TYPES_H_
#define COMMON_SSJOIN_TYPES_H_

#pragma once

#include <cstdint>

typedef int32_t record_t; // record id, size or token
typedef uint8_t overlap_t; // overlap count

struct index_record
{
    record_t key;
    record_t size;
    record_t remaining_tokens;
};

struct record_pair
{
    record_t id_high;
    record_t id_low;
};

#endif
