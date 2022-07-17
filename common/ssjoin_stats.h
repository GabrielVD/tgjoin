#ifndef COMMON_RUNTIME_STATS_H_
#define COMMON_RUNTIME_STATS_H_

#pragma once

#include <ssjoin_types.cuh>
#include <stdio.h>
#include <chrono>
#include <helper_interface.h>

#define NOW() std::chrono::steady_clock::now()
#define TIME_MS(diff) std::chrono::duration<double, std::milli>(diff).count()

enum ssjoin_status
{
    UNDEFINED,
    MEM_ERR,
    SUCCESS
};

struct ssjoin_stats
{
    size_t matrix_size{0};
    size_t matrix_bytesize{0};
    double host2device_ms{0};
    double indexing_ms{0};
    double filtering_ms{0};
    int iterations{0};
    int token_queries{0};
    int index_queries{0};
    record_t token_map_limit{0};
    record_t indexed_entries{0};
    ssjoin_status status{ssjoin_status::UNDEFINED};

    void print(FILE *stream);
};

#endif
