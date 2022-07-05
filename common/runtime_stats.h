#ifndef COMMON_RUNTIME_STATS_H_
#define COMMON_RUNTIME_STATS_H_

#pragma once

#include <stdio.h>
#include <chrono>

#define NOW() std::chrono::steady_clock::now()
#define TIME_MS(diff) std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()

struct dataset_stats
{
    double avg_set_size{.0};
    int cardinality{0};

    void print(FILE *stream)
    {
        fprintf(stream,
                "Avg set size\t%.1lf\n"
                "Cardinality\t%d\n",
                avg_set_size,
                cardinality);
    };
};

enum ssjoin_status
{
    UNDEFINED,
    MEM_ERR,
    SUCCESS
};

struct ssjoin_stats
{
    int host2device_ms{0};
    int indexing_ms{0};
    ssjoin_status status{ssjoin_status::UNDEFINED};

    void print(FILE *stream);
};

#endif
