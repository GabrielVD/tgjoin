#include <ssjoin_staging.h>
#include <helper_cuda.h>
#include <helper_mem.cuh>
#include <sys/stat.h>

#define WIDTH sizeof(record_t)

int check_filesize(const char *pathname, size_t *datacount)
{
    struct stat stats;
    int status{stat(pathname, &stats)};
    if (status != 0) { return status; }

    *datacount = stats.st_size / WIDTH;
    return 0;
}

int load_file(const char *pathname, record_t **ptr, const size_t datacount)
{
    FILE *stream{fopen(pathname, "rb")};
    if (stream == NULL) { return 1; }

    void *buffer;
    checkCudaErrors(cudaMallocHost(&buffer, datacount * WIDTH));

    if (fread(buffer, WIDTH, datacount, stream) != datacount)
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
    for (size_t i = 1; i < info.datacount;)
    {
        ++info.cardinality;
        auto set_size{input[i]};

        if (set_size < last_size
        || i + set_size >= info.datacount)
        {
            return 1;
        }

        last_size = set_size;
        i += set_size + 2;
    }

    info.avg_set_size = (info.datacount - 2 * info.cardinality) / (float)info.cardinality;
    return 0;
}

void input_info::print(FILE *stream)
{
    fprintf(stream,
            "Dataset size" TABS "%ldMB\n"
            "Average set size\t%.1f\n"
            "Cardinality" TABS "%d\n",
            BYTES_R(datacount) / 1000000,
            avg_set_size,
            cardinality);
}
