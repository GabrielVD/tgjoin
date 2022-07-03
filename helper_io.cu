#include <helper_io.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>

int load_file(const char *pathname, uint32_t **dest, size_t *nmemb_dest)
{
    struct stat stats;
    int status{stat(pathname, &stats)};
    if (status != 0)
    {
        return status;
    }

    FILE *handle{fopen(pathname, "rb")};
    if (!handle)
    {
        return 1;
    }

    size_t width{sizeof(uint32_t)};
    size_t nmemb{stats.st_size / width};
    void *buffer{malloc(nmemb * width)};
    if (buffer == NULL)
    {
        fclose(handle);
        return 1;
    }

    if (fread(buffer, width, nmemb, handle) != nmemb)
    {
        free(buffer);
        fclose(handle);
        return 1;
    }

    status = fclose(handle);
    if (status != 0)
    {
        free(buffer);
        return status;
    }

    *dest = (uint32_t *)buffer;
    *nmemb_dest = nmemb;
    return 0;
}

int verify_dataset(const uint32_t *input, size_t input_size, dataset_stats &output)
{
    dataset_stats stats;
    uint32_t last_size{0};
    for (size_t i = 1; i < input_size;)
    {
        ++stats.cardinality;
        auto set_size{input[i]};

        if (set_size < last_size
        || i + set_size >= input_size)
        {
            return 1;
        }

        stats.avg_set_size += set_size;
        last_size = set_size;
        i += set_size + 2;
    }

    stats.avg_set_size /= stats.cardinality;
    output = stats;
    return 0;
}
