#include <ssjoin.h>
#include <ssjoin_shared.h>
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
    uint32_t *buffer, *records_d, *buffer_d;
    size_t buffer_size;

    {
        auto start{NOW()};
        transfer_records_async(
            &buffer, &records_d, buffer_size,
            input, input_size, dstats.cardinality);

        const auto bytes{BYTES_U(buffer_size)};
        checkCudaErrors(cudaMalloc(&buffer_d, bytes));
        checkCudaErrors(cudaMemsetAsync(buffer_d, 0, bytes));

        config = launch_config();

        checkCudaErrors(cudaDeviceSynchronize());
        stats.host2device_ms = TIME_MS(NOW() - start);
    }

    uint32_t *token_map_d;
    {
        auto start{NOW()};
        count_tokens<<<config.count_tokens_grid, config.count_tokens_block>>>(
            records_d, dstats.cardinality, buffer_d, 0.9f);
            
        checkCudaErrors(
            cudaMemcpyAsync(buffer, buffer_d, BYTES_U(2), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());

        stats.token_map_size = buffer[0] + 1;

        checkCudaErrors(cudaMalloc(&token_map_d, BYTES_U(stats.token_map_size)));
        prefix_sum(buffer_d + 2, stats.token_map_size, token_map_d);

        stats.index_entries = buffer[1];
        checkCudaErrors(cudaDeviceSynchronize());

        stats.indexing_ms = TIME_MS(NOW() - start);
    }

    checkCudaErrors(cudaFree(token_map_d));
    checkCudaErrors(cudaFree(buffer_d));
    checkCudaErrors(cudaFree(records_d));
    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
