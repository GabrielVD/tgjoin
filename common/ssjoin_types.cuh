#ifndef COMMON_SSJOIN_TYPES_H_
#define COMMON_SSJOIN_TYPES_H_

#pragma once

#include <cstdint>

typedef int32_t record_t; // record id, size or token

struct index_record
{
    record_t id;
    record_t size;
    record_t remaining_tokens;
};

#endif
