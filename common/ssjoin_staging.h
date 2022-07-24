#ifndef COMMON_SSJOIN_STAGING_H_
#define COMMON_SSJOIN_STAGING_H_

#pragma once

#include <ssjoin_stats.h>
#include <helper_mem.cuh>

struct input_info
{
    const char *pathname;
    size_t datacount{0}, mem_min{0};
    record_t cardinality{0};
    float avg_set_size{.0f};
    float threshold{.0f};
    float overlap_factor{.0f};

    void print(FILE *stream);
};

int check_filesize(const char *pathname, size_t *datacount);
int load_file(const char *pathname, record_t **ptr, size_t datacount);
int verify_dataset(const record_t *input, input_info &);

#endif