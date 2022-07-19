#include <ssjoin_staging.h>
#include <helper_cuda.h>
#include <helper_mem.cuh>
#include <sys/stat.h>

int load_file(const char *pathname, record_t **dest, size_t *nmemb_dest)
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

    size_t width{sizeof(record_t)};
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

    *dest = (record_t *)buffer;
    *nmemb_dest = nmemb;
    return 0;
}

int verify_dataset(const record_t *input, input_info &info)
{
    record_t last_size{0};
    for (size_t i = 1; i < info.data_size;)
    {
        ++info.cardinality;
        auto set_size{input[i]};

        if (set_size < last_size
        || i + set_size >= info.data_size)
        {
            return 1;
        }

        info.avg_set_size += set_size;
        last_size = set_size;
        i += set_size + 2;
    }

    info.avg_set_size /= info.cardinality;
    return 0;
}

void transfer_records_async(
    record_t **records_out,
    record_t **records_d_out,
    size_t &buffer_size,
    const record_t *input,
    size_t input_size,
    int cardinality)
{
    buffer_size = input_size - cardinality + 1;
    const size_t bytes{BYTES_R(buffer_size)};
    checkCudaErrors(cudaMalloc(records_d_out, bytes));
    checkCudaErrors(cudaMallocHost(records_out, bytes));

    record_t *records{*records_out}, *records_d{*records_d_out};
    {
        size_t set_start = cardinality + 1;
        records[0] = set_start;
        for (size_t i{1}, i_input{1}; i_input < input_size; ++i)
        {
            size_t set_size{input[i_input]};
            memcpy(records + set_start, input + i_input + 1, BYTES_R(set_size));

            i_input += set_size + 2;
            set_start += set_size;
            records[i] = set_start;
        }
    }

    checkCudaErrors(cudaMemcpyAsync(records_d, records, bytes, cudaMemcpyHostToDevice));
}
