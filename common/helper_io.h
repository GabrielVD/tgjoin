#ifndef COMMON_HELPER_IO_H_
#define COMMON_HELPER_IO_H_

#pragma once

#include <runtime_stats.h>

int load_file(const char *pathname, uint32_t **dest, size_t *nmemb_dest);
int verify_dataset(const uint32_t *input, size_t input_size, dataset_stats &);

inline int load_dataset(const char *pathname, uint32_t **dataset, size_t *size)
{
    fprintf(stderr, "Reading file %s\n", pathname);

    auto start{NOW()};
    auto status{load_file(pathname, dataset, size)};
    if (status != 0)
    {
        fprintf(stderr, "Error reading file %s\n", pathname);
        return status;
    }
    auto end{NOW()};

    fprintf(stderr, "Loaded %ld bytes in %ldms\n",
            *size * sizeof(uint32_t),
            TIME_MS(end - start));

    return 0;
}

#endif
