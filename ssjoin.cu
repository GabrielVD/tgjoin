#include <ssjoin.h>
#include <ssjoin_staging.h>
#include <ssjoin_index.cuh>
#include <ssjoin_filtering.cuh>
#include <helper_mem.cuh>
#include <helper_cuda.h>
#include <algorithm>
#include <similarity.cuh>

struct kernel_config
{
    launch_params count_tokens;
    launch_params make_index;
    launch_params filter;
};

struct pointers
{
    record_t *buffer, *buffer_d, *records_d, *token_map_d;
    index_record *inverted_index_d;
    uint8_t *overlap_matrix_d;
    size_t buffer_size;
};

static kernel_config get_config()
{
    kernel_config config;

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.count_tokens.grid,
        &config.count_tokens.block,
        count_tokens));

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.make_index.grid,
        &config.make_index.block,
        make_index));

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.filter.grid,
        &config.filter.block,
        filter));

    return config;
}

static void host_to_device(
    const record_t *input,
    const input_info &info,
    pointers &p,
    float &overlap_factor)
{
    transfer_records_async(
        &p.buffer, &p.records_d, p.buffer_size,
        input, info.data_size, info.cardinality);

    const auto bytes{BYTES_U(p.buffer_size)};
    checkCudaErrors(cudaMalloc(&p.buffer_d, bytes));
    checkCudaErrors(cudaMemsetAsync(p.buffer_d, 0, bytes));
    overlap_factor = OVERLAP_FAC(info.threshold);

    checkCudaErrors(cudaDeviceSynchronize());
}

static void indexing(
    ssjoin_stats &stats,
    const kernel_config &config,
    int cardinality,
    pointers &p,
    const float overlap_factor)
{
    count_tokens<<<config.count_tokens.grid, config.count_tokens.block>>>(
        p.records_d,
        cardinality,
        p.buffer_d,
        overlap_factor);

    // copy [token_max, token_count]
    checkCudaErrors(
        cudaMemcpyAsync(p.buffer, p.buffer_d, BYTES_U(2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // limit of starting indexes
    stats.token_map_limit = p.buffer[0] + 1;

    checkCudaErrors(cudaMalloc(&p.token_map_d, BYTES_U(stats.token_map_limit + 1)));
    prefix_sum(p.buffer_d + 2, stats.token_map_limit + 1, p.token_map_d);

    stats.indexed_entries = p.buffer[1];
    checkCudaErrors(cudaMalloc(&p.inverted_index_d, BYTES_INDEX(stats.indexed_entries)));

    make_index<<<config.make_index.grid, config.make_index.block>>>(
        p.records_d,
        cardinality,
        p.token_map_d,
        overlap_factor,
        p.buffer_d + 3, // starting address of token count
        p.inverted_index_d);

    checkCudaErrors(cudaDeviceSynchronize());
}

static void filtering(
    ssjoin_stats &stats,
    const kernel_config &config,
    const input_info &info,
    pointers &p,
    const float overlap_factor)
{
    checkCudaErrors(cudaMemGetInfo(&stats.matrix_bytesize, NULL));
    int id_limit = tri_maxfit((stats.matrix_bytesize - info.mem_min)
                                / sizeof(*p.overlap_matrix_d));
    id_limit = std::min(id_limit, info.cardinality);
    stats.matrix_size = tri_rowstart(id_limit);
    stats.matrix_bytesize = stats.matrix_size * sizeof(*p.overlap_matrix_d);
    checkCudaErrors(cudaMalloc(&p.overlap_matrix_d, stats.matrix_bytesize));

    int id_start{1};
    auto dirty_bytes{stats.matrix_bytesize};
    auto *matrix_tip_d{p.overlap_matrix_d - tri_rowstart(id_start)};
    do
    {
        checkCudaErrors(cudaMemsetAsync(p.overlap_matrix_d, 0, dirty_bytes));
        filter<<<config.filter.grid, config.filter.block>>>(
            p.records_d,
            id_start,
            id_limit,
            p.token_map_d,
            stats.token_map_limit,
            p.inverted_index_d,
            info.threshold,
            overlap_factor,
            matrix_tip_d,
            p.buffer_d);

        ++stats.iterations;
        id_start = id_limit;
        id_limit = tri_maxfit(stats.matrix_size + tri_rowstart(id_start));
        id_limit = std::min(id_limit, info.cardinality);
        matrix_tip_d = p.overlap_matrix_d - tri_rowstart(id_start);
        dirty_bytes = ((matrix_tip_d + tri_rowstart(id_limit)) - p.overlap_matrix_d)
            * sizeof(*p.overlap_matrix_d);
        checkCudaErrors(cudaDeviceSynchronize());
    } while (id_start < info.cardinality);
}

ssjoin_stats run_join(const record_t *input, input_info info)
{
    ssjoin_stats stats;
    kernel_config config{get_config()};
    pointers p;
    float overlap_factor;

    {
        auto start{NOW()};
        host_to_device(input, info, p, overlap_factor);
        stats.host2device_ms = TIME_MS(NOW() - start);
    }

    {
        auto start{NOW()};
        indexing(stats, config, info.cardinality, p, overlap_factor);
        stats.indexing_ms = TIME_MS(NOW() - start);
    }

    {
        checkCudaErrors(cudaMemset(p.buffer_d, 0, BYTES_U(2)));
        auto start{NOW()};
        filtering(stats, config, info, p, overlap_factor);
        stats.filtering_ms = TIME_MS(NOW() - start);
        checkCudaErrors(cudaMemcpy(p.buffer, p.buffer_d, BYTES_U(2), cudaMemcpyDeviceToHost));
        stats.token_probes = p.buffer[0];
        stats.index_probes = p.buffer[1];
    }

    checkCudaErrors(cudaFree(p.overlap_matrix_d));
    checkCudaErrors(cudaFree(p.inverted_index_d));
    checkCudaErrors(cudaFree(p.token_map_d));
    checkCudaErrors(cudaFree(p.buffer_d));
    checkCudaErrors(cudaFree(p.records_d));
    stats.status = ssjoin_status::SUCCESS;
    return stats;
}
