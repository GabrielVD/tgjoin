#ifndef COMMON_SSJOIN_STAGING_H_
#define COMMON_SSJOIN_STAGING_H_

#pragma once

#include <ssjoin_stats.h>

struct input_info
{
    size_t data_size{0}, mem_min{0};
    int cardinality{0};
    float avg_set_size{.0f}, threshold{.0f};

    void print(FILE *stream)
    {
        fprintf(stream,
                "Average set size\t%.1f\n"
                "Cardinality" TABS "%d\n",
                avg_set_size,
                cardinality);
    };
};

int load_file(const char *pathname, uint32_t **dest, size_t *nmemb_dest);
int verify_dataset(const uint32_t *input, input_info &);

void transfer_records_async(
    uint32_t **records_out,
    uint32_t **records_d_out,
    size_t &buffer_size,
    const uint32_t *input,
    size_t input_size,
    int cardinality);

inline int load_dataset(const char *pathname, uint32_t **dataset, size_t *size)
{
    fprintf(stderr, "Document: %s\n", pathname);

    auto start{NOW()};
    auto status{load_file(pathname, dataset, size)};
    if (status != 0)
    {
        fprintf(stderr, "Error reading file %s\n", pathname);
        return status;
    }
    auto end{NOW()};

    fprintf(stderr, "Read %ldMB in %ldms\n",
            (*size * sizeof(uint32_t)) / 1000000,
            TIME_MS(end - start));

    return 0;
}

#endif