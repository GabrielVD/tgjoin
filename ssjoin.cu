#include <ssjoin.h>
#include <helper_mem.h>
#include <helper_cuda.h>

int transfer_records(
    uint32_t **records_out,
    uint32_t **records_d_out,
    const uint32_t *input,
    size_t input_size,
    int cardinality,
    ssjoin_stats &stats)
{
    uint32_t *records, *records_d;
    const size_t records_size{input_size - cardinality + 1};

    auto start{NOW()};
    checkCudaErrors(cudaMalloc(&records_d, BYTES_U(records_size)));

    records = MALLOC_U(records_size);
    if (records == NULL)
    {
        stats.status = ssjoin_status::MEM_ERR;
        checkCudaErrors(cudaFree(records_d));
        return 1;
    }

    {
        size_t set_start = cardinality + 1;
        records[0] = set_start;
        for (size_t i{1}, i_input{1}; i_input < input_size; ++i)
        {
            size_t set_size{input[i_input]};
            memcpy(records + set_start, input + i_input + 1, BYTES_U(set_size));

            i_input += set_size + 2;
            set_start += set_size;
            records[i] = set_start;
        }
    }

    checkCudaErrors(cudaMemcpyAsync(
        records_d, records,
        BYTES_U(records_size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaStreamSynchronize(NULL));

    auto end{NOW()};
    stats.host2device_ms = TIME_MS(end - start);
    *records_out = records;
    *records_d_out = records_d;
    return 0;
}

ssjoin_stats run_join(const uint32_t *input, size_t input_size, dataset_stats dstats)
{
    ssjoin_stats stats;
    uint32_t *records, *records_d;

    if (transfer_records(&records, &records_d, input, input_size,
                         dstats.cardinality, stats) == 0)
    {
        checkCudaErrors(cudaFree(records_d));
        free(records);
    }

    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
