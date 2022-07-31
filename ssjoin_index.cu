#include <ssjoin_index.cuh>
#include <helper_cuda.h>
#include <similarity.cuh>

__global__ void count_tokens(
    const record_t* __restrict__ record_map_d,
    const record_t cardinality,
    record_t* __restrict__ count_d,
    const float overlap_factor)
{
    const record_t stride = STRIDE();
    record_t token_max{0}, token_count{0};
    count_d += 3; // reserve 3 cells for [token_max, token_count, 0]

    for (record_t idx = IDX(); idx < cardinality; idx += stride)
    {
        record_t start{record_map_d[idx]};
        const record_t size{
            index_prefix_size_d(
                record_map_d[idx + 1] - 2 - start,
                overlap_factor)};
        token_count += size;
        const record_t end{start + size};
        do
        {
            const auto token{record_map_d[start]};
            token_max = token_max < token ? token : token_max;
            atomicAdd(count_d + token, 1);
        } while (++start < end);
    }
    atomicMax(count_d - 3, token_max);
    atomicAdd(count_d - 2, token_count);
}

__global__ void make_index(
    const record_t* __restrict__ record_map_d,
    const record_t cardinality,
    const record_t* __restrict__ token_map_d,
    const float overlap_factor,
    record_t* __restrict__ count_d,
    index_record* __restrict__ inverted_index_d)
{
    const record_t stride = STRIDE();
    index_record record;
    for (record.key = IDX(); record.key < cardinality; record.key += stride)
    {
        auto start{record_map_d[record.key]};
        record.size = record_map_d[record.key + 1] - 2 - start;
        record.remaining_tokens = record.size;
        const auto end{start + index_prefix_size_d(record.size, overlap_factor)};
        
        for (; start < end; ++start)
        {
            const auto token{record_map_d[start]};
            --record.remaining_tokens;
            const auto head{token_map_d[token] + atomicSub(count_d + token, 1) - 1};
            inverted_index_d[head] = record;
        }
    }
}
