#ifndef COMMON_RUNTIME_STATS_H_
#define COMMON_RUNTIME_STATS_H_

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

struct ssjoin_stats
{
};

#endif
