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

struct pointers_t
{
    byte_t *pool_d, *pool_limit_d;
    record_t *buffer;
    record_t *record_map_d;
    record_t *buffer_d;
    record_t *token_map_d;
    index_record *index_d;
    overlap_t *overlap_matrix_d;
};

struct streams_t
{
    cudaStream_t a, b;
};

struct joinstate_t
{
    pointers_t ptr;
    size_t overlap_capacity;
    ssjoin_stats stats;
    streams_t stream;
    kernel_config config;
    float overlap_factor{.0f};
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

    // checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
    //     &config.filter.grid,
    //     &config.filter.block,
    //     filter));

    return config;
}

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
    record_t token_start = info.cardinality + 3;
    record_map[0] = token_start;
    record_t size_index{1};
    for (int i = 1; i <= info.cardinality; ++i)
    {
        const record_t step{state.ptr.buffer[size_index] + 2};
        token_start += step;
        size_index += step;
        record_map[i] = token_start;
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
    // pool layout
    // [record_map; buffer; token_map; inverted_index; overlap_matrix]
    state.stats.pool_size = pool_size(info);
    checkCudaErrors(cudaMalloc(&state.ptr.pool_d, state.stats.pool_size));

    {
        state.ptr.record_map_d = (record_t*)state.ptr.pool_d;
        record_t *dataset_d = state.ptr.record_map_d + (info.cardinality + 1);
        checkCudaErrors(cudaMemcpyAsync(
            dataset_d,
            state.ptr.buffer,
            BYTES_R(info.datacount),
            cudaMemcpyHostToDevice,
            state.stream.a));

        state.ptr.buffer_d = aligned_up<record_t, 16>(dataset_d + info.datacount);
    }

    state.ptr.pool_limit_d = state.ptr.pool_d + state.stats.pool_size;
    checkCudaErrors(cudaMemsetAsync(
        state.ptr.buffer_d,
        0,
        byte_diff(state.ptr.pool_limit_d, state.ptr.buffer_d),
        state.stream.b));

    record_t *record_map;
    {
        const size_t size = BYTES_R(info.cardinality + 1);
        record_map = (record_t *)malloc(size);
        map_records_async(record_map, size, state, info, state.stream.a);
    }
    
    state.ptr.token_map_d = aligned_up<record_t, 16>(state.ptr.buffer_d + info.datacount);
    state.overlap_factor = OVERLAP_FAC(info.threshold);

    checkCudaErrors(cudaDeviceSynchronize());
    free(record_map);
}

static void indexing(joinstate_t &state, const input_info &info)
{
    count_tokens<<<state.config.count_tokens.grid, state.config.count_tokens.block>>>(
        state.ptr.record_map_d,
        info.cardinality,
        state.ptr.buffer_d,
        state.overlap_factor);

    // copy [token_max, token_count]
    checkCudaErrors(
        cudaMemcpyAsync(
            state.ptr.buffer,
            state.ptr.buffer_d,
            BYTES_R(2),
            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    state.stats.token_map_limit = state.ptr.buffer[0] + 1;
    state.stats.indexed_entries = state.ptr.buffer[1];

    {
        const auto token_map_datacount = state.stats.token_map_limit + 1;

        state.ptr.index_d = (index_record*)aligned_up<record_t, 16>(
            state.ptr.token_map_d + token_map_datacount);

        state.ptr.overlap_matrix_d = (overlap_t*)aligned_up<index_record, 16>(
            state.ptr.index_d + state.stats.indexed_entries);

        state.overlap_capacity = byte_diff(state.ptr.pool_limit_d, state.ptr.overlap_matrix_d);
        state.overlap_capacity /= sizeof(overlap_t);

        prefix_sum(
            state.ptr.buffer_d + 2, // starting address of token count prefixed with a 0
            token_map_datacount,
            state.ptr.token_map_d);
    }

    make_index<<<state.config.make_index.grid, state.config.make_index.block, 0, state.stream.b>>>(
        state.ptr.record_map_d,
        info.cardinality,
        state.ptr.token_map_d,
        state.overlap_factor,
        state.ptr.buffer_d + 3, // starting address of token count
        state.ptr.index_d);

    checkCudaErrors(cudaDeviceSynchronize());
}

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
    state.config = get_config();

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

    {
        auto start{NOW()};
        indexing(state, info);
        state.stats.indexing_ms = TIME_MS(NOW() - start);
    }

    // {
    //     auto start{NOW()};
    //     filtering(stats, config, info, p, overlap_factor);
    //     stats.filtering_ms = TIME_MS(NOW() - start);
    // }
    
    checkCudaErrors(cudaStreamDestroy(state.stream.a));
    checkCudaErrors(cudaStreamDestroy(state.stream.b));
    checkCudaErrors(cudaFree(state.ptr.pool_d));
    checkCudaErrors(cudaFreeHost(state.ptr.buffer));
    state.stats.status = ssjoin_status::SUCCESS;
    return state.stats;
}
