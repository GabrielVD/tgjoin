#include <ssjoin_shared.h>
#include <ssjoin.h>
#include <ssjoin_index.cuh>
#include <helper_mem.h>
#include <helper_cuda.h>

struct ssjoin_lconfig
{
    int count_tokens_grid;
    int count_tokens_block;
};

static ssjoin_lconfig launch_config()
{
    ssjoin_lconfig config;

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.count_tokens_grid,
        &config.count_tokens_block,
        count_tokens));

    return config;
}

ssjoin_stats run_join(const uint32_t *input, size_t input_size, dataset_stats dstats)
{
    ssjoin_stats stats;
    ssjoin_lconfig config;
    size_t buffer_size;
    uint32_t *records, *records_d, *results_d;

    {
        auto start{NOW()};
        transfer_records_async(
            &records, &records_d, buffer_size,
            input, input_size, dstats.cardinality);

        const auto bytes{BYTES_U(buffer_size)};
        checkCudaErrors(cudaMalloc(&results_d, bytes));
        checkCudaErrors(cudaMemsetAsync(results_d, 0, bytes));
        config = launch_config();

        checkCudaErrors(cudaDeviceSynchronize());
        stats.host2device_ms = TIME_MS(NOW() - start);
    }

    {
        auto start{NOW()};
        count_tokens<<<config.count_tokens_grid, config.count_tokens_block>>>(
            records_d, dstats.cardinality, results_d, 0.9f);

        checkCudaErrors(cudaDeviceSynchronize());
        stats.indexing_ms = TIME_MS(NOW() - start);
    }

    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
