#ifndef COMMON_SSJOIN_H_
#define COMMON_SSJOIN_H_

#include <runtime_stats.h>

ssjoin_stats run_join(const uint32_t *input, size_t input_size);

#endif