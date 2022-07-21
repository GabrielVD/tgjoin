#include <ssjoin_staging.h>
#include <helper_cuda.h>
#include <helper_mem.cuh>
#include <sys/stat.h>

#define WIDTH sizeof(record_t)

int check_filesize(const char *pathname, size_t *size)
{
    struct stat stats;
    int status{stat(pathname, &stats)};
    if (status != 0) { return status; }

    *size = stats.st_size / WIDTH;
    return 0;
}

int load_file(const char *pathname, record_t **ptr, const size_t size)
{
    FILE *stream{fopen(pathname, "rb")};
    if (stream == NULL) { return 1; }

    void *buffer;
    checkCudaErrors(cudaMallocHost(&buffer, size * WIDTH));

    if (fread(buffer, WIDTH, size, stream) != size)
    {
        checkCudaErrors(cudaFreeHost(buffer));
        fclose(stream);
        return 1;
    }

    int status = fclose(stream);
    if (status != 0)
    {
        checkCudaErrors(cudaFreeHost(buffer));
        return status;
    }

    *ptr = (record_t *)buffer;
    return 0;
}

int verify_dataset(const record_t *input, input_info &info)
{
    info.cardinality = 0;

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

        last_size = set_size;
        i += set_size + 2;
    }

    info.avg_set_size = (info.data_size - 2 * info.cardinality) / (float)info.cardinality;
    return 0;
}

void transfer_input_async(
    const record_t *input,
    const input_info &info,
    void *buffer_d)
{
    // const size_t bytes{BYTES_R(buffer_size)};
    // checkCudaErrors(cudaMalloc(records_d_out, bytes));
    // checkCudaErrors(cudaMallocHost(records_out, bytes));

    // checkCudaErrors(cudaMemcpyAsync(records_d, records, bytes, cudaMemcpyHostToDevice));
}
