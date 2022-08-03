#include <ssjoin.h>
#include <ssjoin_staging.h>
#include <ssjoin_index.cuh>
#include <ssjoin_filtering.cuh>
#include <ssjoin_verify.cuh>
#include <helper_mem.cuh>
#include <helper_cuda.h>
#include <algorithm>
#include <similarity.cuh>

struct kernel_config
{
    launch_params count_tokens;
    launch_params make_index;
    launch_params filter;
    launch_params_smem verify;
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

    checkCudaErrors(cudaFuncSetAttribute(
        count_tokens,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        0));
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.count_tokens.grid,
        &config.count_tokens.block,
        count_tokens));

    checkCudaErrors(cudaFuncSetAttribute(
        make_index,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        0));
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.make_index.grid,
        &config.make_index.block,
        make_index));

    checkCudaErrors(cudaFuncSetAttribute(
        filter,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        0));
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &config.filter.grid,
        &config.filter.block,
        filter));

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &config.verify.grid,
        &config.verify.block,
        verify,
        verifyBlockSizeToDynamicSMemSize));
    config.verify.smem = verifyBlockSizeToDynamicSMemSize(config.verify.block);

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
        state.ptr.buffer_d = aligned_up<record_t, CACHE_LINE>(dataset_d + info.datacount);
        state.ptr.pool_limit_d = state.ptr.pool_d + state.stats.pool_size;

        checkCudaErrors(cudaMemsetAsync(
            state.ptr.buffer_d,
            0,
            byte_diff(state.ptr.pool_limit_d, state.ptr.buffer_d),
            state.stream.a));

        checkCudaErrors(cudaMemcpyAsync(
            dataset_d,
            state.ptr.buffer,
            BYTES_R(info.datacount),
            cudaMemcpyHostToDevice,
            state.stream.b));
    }

    record_t *record_map;
    {
        const size_t size = BYTES_R(info.cardinality + 1);
        record_map = (record_t *)malloc(size);
        map_records_async(record_map, size, state, info, state.stream.b);
    }
    
    state.ptr.token_map_d = aligned_up<record_t, CACHE_LINE>(state.ptr.buffer_d + info.datacount);
    state.overlap_factor = OVERLAP_FAC(info.threshold);

    checkCudaErrors(cudaDeviceSynchronize());
    free(record_map);
}

static void counting(joinstate_t &state, const input_info &info)
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

    checkCudaErrors(cudaMemsetAsync(state.ptr.buffer_d, 0, 2*sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    state.stats.token_map_limit = state.ptr.buffer[0] + 1;
    state.stats.indexed_entries = state.ptr.buffer[1];
}

static void indexing(joinstate_t &state, const input_info &info)
{
    counting(state, info);

    {
        const auto token_map_datacount = state.stats.token_map_limit + 1;

        state.ptr.index_d = (index_record*)aligned_up<record_t, CACHE_LINE>(
            state.ptr.token_map_d + token_map_datacount);

        prefix_sum(
            state.ptr.buffer_d + 2, // starting address of token count prefixed with a 0
            token_map_datacount,
            state.ptr.token_map_d);
    }

    make_index<<<state.config.make_index.grid, state.config.make_index.block>>>(
        state.ptr.record_map_d,
        info.cardinality,
        state.ptr.token_map_d,
        state.overlap_factor,
        state.ptr.buffer_d + 3, // starting address of token count
        state.ptr.index_d);

    state.ptr.overlap_matrix_d = (overlap_t*)aligned_up<index_record, CACHE_LINE>(
        state.ptr.index_d + state.stats.indexed_entries);

    state.overlap_capacity = byte_diff(state.ptr.pool_limit_d, state.ptr.overlap_matrix_d);
    state.overlap_capacity /= sizeof(overlap_t);

    checkCudaErrors(cudaDeviceSynchronize());
}

static record_t find_key_limit(size_t capacity)
{
    return tri_maxfit(OVERLAP_PACK_SIZE * (capacity / OVERLAP_PACK_SIZE));
}

static int pack_count(record_t key_limit, size_t overlap_offset)
{
    return (tri_rowstart(key_limit) - overlap_offset + OVERLAP_PACK_SIZE - 1) / OVERLAP_PACK_SIZE;
}

static void filtering(joinstate_t &state, const input_info &info)
{
    record_t key_limit = find_key_limit(state.overlap_capacity);
    key_limit = std::min(key_limit, info.cardinality);
    
    record_t key_start = 1;
    size_t overlap_offset = 0;
    do
    {
        filter<<<state.config.filter.grid, state.config.filter.block>>>(
            state.ptr.record_map_d,
            key_start,
            key_limit,
            state.ptr.token_map_d,
            state.stats.token_map_limit,
            state.ptr.index_d,
            info.threshold,
            state.overlap_factor,
            state.ptr.overlap_matrix_d,
            overlap_offset,
            (int*)state.ptr.buffer_d);
        
        verify<<<
        state.config.verify.grid,
        state.config.verify.block,
        state.config.verify.smem>>>(
            ((int*)state.ptr.buffer_d) + 1,
            (overlap_pack*)state.ptr.overlap_matrix_d,
            pack_count(key_limit, overlap_offset));

        ++state.stats.iterations;
        key_start = key_limit;
        overlap_offset = tri_rowstart(key_limit);
        key_limit = find_key_limit(state.overlap_capacity + overlap_offset);
        key_limit = std::min(key_limit, info.cardinality);
    } while (key_start < info.cardinality);

        checkCudaErrors(
            cudaMemcpyAsync(
                state.ptr.buffer,
                state.ptr.buffer_d,
                2*sizeof(int),
                cudaMemcpyDeviceToHost));
        
        checkCudaErrors(cudaDeviceSynchronize());

        printf("Filtered: %d\nVerified: %d\n", state.ptr.buffer[0], state.ptr.buffer[1]);
}

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

    {
        auto start{NOW()};
        filtering(state, info);
        state.stats.filtering_ms = TIME_MS(NOW() - start);
    }
    
    checkCudaErrors(cudaStreamDestroy(state.stream.a));
    checkCudaErrors(cudaStreamDestroy(state.stream.b));
    checkCudaErrors(cudaFree(state.ptr.pool_d));
    checkCudaErrors(cudaFreeHost(state.ptr.buffer));
    state.stats.status = ssjoin_status::SUCCESS;
    return state.stats;
}
