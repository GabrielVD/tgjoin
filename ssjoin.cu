#include <ssjoin.h>
#include <ssjoin_staging.h>
#include <ssjoin_index.cuh>
#include <ssjoin_filtering.cuh>
#include <helper_mem.cuh>
#include <helper_cuda.h>
#include <algorithm>
#include <similarity.cuh>

// struct kernel_config
// {
//     launch_params count_tokens;
//     launch_params make_index;
//     launch_params filter;
// };

struct pointers_t
{
    byte_t *pool_d, *pool_limit_d;
    record_t *buffer, *dataset_d, *record_map_d, *buffer_d;
};

struct streams_t
{
    cudaStream_t a, b;
};

struct joinstate_t
{
    pointers_t ptr;
    ssjoin_stats stats;
    streams_t stream;
};

// struct pointers
// {
//     record_t *buffer, *buffer_d, *records_d, *token_map_d;
//     index_record *inverted_index_d;
//     uint8_t *overlap_matrix_d;
//     size_t buffer_size;
// };

// static kernel_config get_config()
// {
//     kernel_config config;

//     checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
//         &config.count_tokens.grid,
//         &config.count_tokens.block,
//         count_tokens));

//     checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
//         &config.make_index.grid,
//         &config.make_index.block,
//         make_index));

//     checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
//         &config.filter.grid,
//         &config.filter.block,
//         filter));

//     return config;
// }

static int file_read(ssjoin_stats &stats, input_info &info, record_t **buffer)
{
    if (check_filesize(info.pathname, &info.datacount) != 0
    || load_file(info.pathname, buffer, info.datacount) != 0)
    {
        stats.status = ssjoin_status::IO_ERR;
        return 1;
    }

    if (verify_dataset(*buffer, info) != 0)
    {
        stats.status = ssjoin_status::FORMAT_ERR;
        checkCudaErrors(cudaFreeHost(*buffer));
        *buffer = NULL;
        return 1;
    }

    return 0;
}

static size_t pool_size(const input_info &info)
{
    size_t size;
    checkCudaErrors(cudaMemGetInfo(&size, NULL));
    size -= info.mem_min;

    const size_t overlap_size{BYTES_O(tri_rowstart(info.cardinality + 1))};
    if (overlap_size < size)
    {
        size = std::min(
            size,
            2 * BYTES_R(info.datacount)
                + BYTES_R(info.cardinality)
                + BYTES_I(info.cardinality * (size_t)info.avg_set_size)
                + overlap_size);
    }

    return size;
}

static void map_records_async(
    record_t *record_map,
    size_t size,
    const joinstate_t &state,
    const input_info &info,
    cudaStream_t stream)
{
    record_t record_index{1};
    for (int i = 0; i < info.cardinality; i++)
    {
        record_map[i] = record_index;
        record_index += state.ptr.buffer[record_index] + 2;
    }

    checkCudaErrors(cudaMemcpyAsync(
        state.ptr.record_map_d,
        record_map,
        size,
        cudaMemcpyHostToDevice,
        stream));
}

static void host_to_device(joinstate_t &state, input_info &info)
{
    state.stats.pool_size = pool_size(info);
    checkCudaErrors(cudaMalloc(&state.ptr.pool_d, state.stats.pool_size));

    size_t input_bytes = BYTES_R(info.datacount);
    checkCudaErrors(cudaMemcpyAsync(
        state.ptr.pool_d,
        state.ptr.buffer,
        input_bytes,
        cudaMemcpyHostToDevice,
        state.stream.a));

    state.ptr.record_map_d = (record_t*)(state.ptr.pool_d + input_bytes);
    state.ptr.pool_limit_d = state.ptr.pool_d + state.stats.pool_size;
    state.ptr.buffer_d = state.ptr.record_map_d + info.cardinality;

    const size_t cardinality_bytes = BYTES_R(info.cardinality);
    checkCudaErrors(cudaMemsetAsync(
        state.ptr.buffer_d,
        0,
        input_bytes - 2 * cardinality_bytes,
        state.stream.b));

    record_t *record_map = (record_t *)malloc(cardinality_bytes);
    map_records_async(record_map, cardinality_bytes, state, info, state.stream.a);
    
    info.overlap_factor = OVERLAP_FAC(info.threshold);

    state.ptr.dataset_d = (record_t*)state.ptr.pool_d;
    checkCudaErrors(cudaDeviceSynchronize());
    free(record_map);
}

// static void indexing(
//     ssjoin_stats &stats,
//     const kernel_config &config,
//     record_t cardinality,
//     pointers &p,
//     const float overlap_factor)
// {
//     count_tokens<<<config.count_tokens.grid, config.count_tokens.block>>>(
//         p.records_d,
//         cardinality,
//         p.buffer_d,
//         overlap_factor);

//     // copy [token_max, token_count]
//     checkCudaErrors(
//         cudaMemcpyAsync(p.buffer, p.buffer_d, BYTES_R(2), cudaMemcpyDeviceToHost));
//     checkCudaErrors(cudaDeviceSynchronize());

//     // limit of starting indexes
//     stats.token_map_limit = p.buffer[0] + 1;

//     checkCudaErrors(cudaMalloc(&p.token_map_d, BYTES_R(stats.token_map_limit + 1)));
//     prefix_sum(p.buffer_d + 2, stats.token_map_limit + 1, p.token_map_d);

//     stats.indexed_entries = p.buffer[1];
//     checkCudaErrors(cudaMalloc(&p.inverted_index_d, BYTES_INDEX(stats.indexed_entries)));

//     make_index<<<config.make_index.grid, config.make_index.block>>>(
//         p.records_d,
//         cardinality,
//         p.token_map_d,
//         overlap_factor,
//         p.buffer_d + 3, // starting address of token count
//         p.inverted_index_d);

//     checkCudaErrors(cudaDeviceSynchronize());
// }

// static void filtering(
//     ssjoin_stats &stats,
//     const kernel_config &config,
//     const input_info &info,
//     pointers &p,
//     const float overlap_factor)
// {
//     checkCudaErrors(cudaMemGetInfo(&stats.matrix_bytesize, NULL));
//     record_t id_limit = tri_maxfit((stats.matrix_bytesize - info.mem_min)
//                                 / sizeof(*p.overlap_matrix_d));
//     id_limit = std::min(id_limit, info.cardinality);
//     stats.matrix_size = tri_rowstart(id_limit);
//     stats.matrix_bytesize = stats.matrix_size * sizeof(*p.overlap_matrix_d);
//     checkCudaErrors(cudaMalloc(&p.overlap_matrix_d, stats.matrix_bytesize));

//     record_t id_start{1};
//     auto dirty_bytes{stats.matrix_bytesize};
//     auto *matrix_tip_d{p.overlap_matrix_d - tri_rowstart(id_start)};
//     do
//     {
//         checkCudaErrors(cudaMemsetAsync(p.overlap_matrix_d, 0, dirty_bytes));
//         filter<<<config.filter.grid, config.filter.block>>>(
//             p.records_d,
//             id_start,
//             id_limit,
//             p.token_map_d,
//             stats.token_map_limit,
//             p.inverted_index_d,
//             info.threshold,
//             overlap_factor,
//             matrix_tip_d,
//             p.buffer_d);

//         ++stats.iterations;
//         id_start = id_limit;
//         id_limit = tri_maxfit(stats.matrix_size + tri_rowstart(id_start));
//         id_limit = std::min(id_limit, info.cardinality);
//         matrix_tip_d = p.overlap_matrix_d - tri_rowstart(id_start);
//         dirty_bytes = ((matrix_tip_d + tri_rowstart(id_limit)) - p.overlap_matrix_d)
//             * sizeof(*p.overlap_matrix_d);
//         checkCudaErrors(cudaDeviceSynchronize());
//     } while (id_start < info.cardinality);
// }

ssjoin_stats run_join(input_info info)
{
    joinstate_t state;

    {
        auto start{NOW()};
        if (file_read(state.stats, info, &state.ptr.buffer) != 0)
        {
            return state.stats;
        }
        state.stats.read_ms = TIME_MS(NOW() - start);
        info.print(stderr);
    }

    {
        auto start{NOW()};
        checkCudaErrors(cudaStreamCreate(&state.stream.a));
        checkCudaErrors(cudaStreamCreate(&state.stream.b));
        host_to_device(state, info);
        state.stats.host2device_ms = TIME_MS(NOW() - start);
    }

    // kernel_config config{get_config()};
    // pointers p;
    // float overlap_factor;

    // {
    //     auto start{NOW()};
    //     host_to_device(input, info, p, overlap_factor);
    //     stats.host2device_ms = TIME_MS(NOW() - start);
    // }

    // {
    //     auto start{NOW()};
    //     indexing(stats, config, info.cardinality, p, overlap_factor);
    //     stats.indexing_ms = TIME_MS(NOW() - start);
    // }

    // {
    //     checkCudaErrors(cudaMemset(p.buffer_d, 0, BYTES_R(2)));
    //     auto start{NOW()};
    //     filtering(stats, config, info, p, overlap_factor);
    //     stats.filtering_ms = TIME_MS(NOW() - start);
    //     checkCudaErrors(cudaMemcpy(p.buffer, p.buffer_d, BYTES_R(2), cudaMemcpyDeviceToHost));
    //     stats.token_probes = p.buffer[0];
    //     stats.index_probes = p.buffer[1];
    // }
    
    checkCudaErrors(cudaFree(state.ptr.pool_d));
    checkCudaErrors(cudaFreeHost(state.ptr.buffer));
    state.stats.status = ssjoin_status::SUCCESS;
    return state.stats;
}
