#ifndef COMMON_RUNTIME_STATS_H_
#define COMMON_RUNTIME_STATS_H_

#pragma once

#include <stdio.h>
#include <chrono>
#include <helper_interface.h>

#define NOW() std::chrono::steady_clock::now()
#define TIME_MS(diff) std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()

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
    int host2device_ms{0};
    int indexing_ms{0};
    int filtering_ms{0};
    int iterations{0};
    int token_queries{0};
    int index_queries{0};
    uint32_t token_map_limit{0};
    uint32_t indexed_entries{0};
    ssjoin_status status{ssjoin_status::UNDEFINED};

    void print(FILE *stream);
};

#endif
