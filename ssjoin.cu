#include <ssjoin.h>
#include <ssjoin_shared.h>
#include <ssjoin_index.cuh>
#include <helper_mem.h>
#include <helper_cuda.h>

struct ssjoin_config
{
    launch_config count_tokens;
    launch_config make_index;
    // launch_config sort_index;
};

struct pointers
{
    uint32_t *buffer, *buffer_d, *records_d, *token_map_d;
    index_record *inverted_index_d;
};

static ssjoin_config launch_config()
{
    ssjoin_config config;

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.count_tokens.grid,
        &config.count_tokens.block,
        count_tokens));

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.make_index.grid,
        &config.make_index.block,
        make_index));

    // checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
    //     &config.sort_index.grid,
    //     &config.sort_index.block,
    //     sort_index));

    return config;
}

ssjoin_stats run_join(const uint32_t *input, size_t input_size, dataset_stats dstats)
{
    ssjoin_stats stats;
    ssjoin_config config{launch_config()};
    pointers p;
    size_t buffer_size;

    {
        auto start{NOW()};
        transfer_records_async(
            &p.buffer, &p.records_d, buffer_size,
            input, input_size, dstats.cardinality);

        const auto bytes{BYTES_U(buffer_size)};
        checkCudaErrors(cudaMalloc(&p.buffer_d, bytes));
        checkCudaErrors(cudaMemsetAsync(p.buffer_d, 0, bytes));
        
        checkCudaErrors(cudaDeviceSynchronize());
        stats.host2device_ms = TIME_MS(NOW() - start);
    }

    {
        auto start{NOW()};
        count_tokens<<<config.count_tokens.grid, config.count_tokens.block>>>(
            p.records_d, dstats.cardinality, p.buffer_d, 0.9f);
        
        // copy [token_max, token_count]
        checkCudaErrors(
            cudaMemcpyAsync(p.buffer, p.buffer_d, BYTES_U(2), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        stats.token_map_size = p.buffer[0] + 2;

        checkCudaErrors(cudaMalloc(&p.token_map_d, BYTES_U(stats.token_map_size)));
        prefix_sum(p.buffer_d + 2, stats.token_map_size, p.token_map_d);

        stats.indexed_entries = p.buffer[1];
        checkCudaErrors(cudaMalloc(&p.inverted_index_d, BYTES_INDEX(stats.indexed_entries)));

        make_index<<<config.make_index.grid, config.make_index.block>>>(
            p.records_d,
            dstats.cardinality,
            p.token_map_d,
            0.9f,
            p.buffer_d + 3, // starting address of token count
            p.inverted_index_d);

        // sort_index<<<config.sort_index.grid, config.sort_index.block>>>(
        //     p.token_map_d,
        //     stats.token_map_size,
        //     p.inverted_index_d);
        checkCudaErrors(cudaDeviceSynchronize());

        stats.indexing_ms = TIME_MS(NOW() - start);
    }

    checkCudaErrors(cudaFree(p.token_map_d));
    checkCudaErrors(cudaFree(p.buffer_d));
    checkCudaErrors(cudaFree(p.records_d));
    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
