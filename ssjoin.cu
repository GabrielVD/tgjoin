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

    uint32_t *index_d;
    {
        auto start{NOW()};
        count_tokens<<<config.count_tokens_grid, config.count_tokens_block>>>(
            records_d, dstats.cardinality, buffer_d, 0.9f);
            
        checkCudaErrors(
            cudaMemcpyAsync(buffer, buffer_d, BYTES_U(1), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());

        stats.max_indexed_token = buffer[0];
        const auto size{stats.max_indexed_token + 1};

        checkCudaErrors(cudaMalloc(&index_d, BYTES_U(size)));
        prefix_sum(buffer_d + 1, size, index_d);
        checkCudaErrors(cudaDeviceSynchronize());

        stats.indexing_ms = TIME_MS(NOW() - start);
    }

    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
