#ifndef COMMON_SSJOIN_STAGING_H_
#define COMMON_SSJOIN_STAGING_H_

#pragma once

#include <ssjoin_stats.h>

struct input_info
{
    const char *pathname;
    size_t data_size{0}, mem_min{0};
    record_t cardinality{0};
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

int check_filesize(const char *pathname, size_t *size);
int load_file(const char *pathname, record_t **ptr, size_t size);
int verify_dataset(const record_t *input, input_info &);

// inline int load_dataset(const char *pathname, record_t **dataset, size_t *size)
// {
//     fprintf(stderr, "Document: %s\n", pathname);

//     auto start{NOW()};
//     auto status{load_file(pathname, dataset, size)};
//     if (status != 0)
//     {
//         fprintf(stderr, "Error reading file %s\n", pathname);
//         return status;
//     }
//     auto end{NOW()};

//     fprintf(stderr, "Read %ldMB in %.3lfms\n",
//             (*size * sizeof(record_t)) / 1000000,
//             TIME_MS(end - start));

//     return 0;
// }

#endif