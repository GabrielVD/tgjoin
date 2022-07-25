#include <ssjoin_index.cuh>
#include <helper_cuda.h>
#include <similarity.cuh>

__global__ void count_tokens(
    const record_t *record_map_d,
    const record_t cardinality,
    record_t *count_d,
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
    const record_t *records_d,
    const record_t cardinality,
    const record_t *token_map_d,
    const float overlap_factor,
    record_t *count_d,
    index_record *inverted_index_d)
{
    const record_t stride = STRIDE();
    index_record record;
    for (record.id = IDX(); record.id < cardinality; record.id += stride)
    {
        auto start{records_d[record.id]};
        record.size = records_d[record.id + 1] - start;
        const auto end{start + index_prefix_size_d(record.size, overlap_factor)};
        
        while (start < end)
        {
            const auto token{records_d[start]};
            ++start;
            record.remaining_tokens = record.size - start;
            const auto head{token_map_d[token] + atomicSub(count_d + token, 1)};
            inverted_index_d[head] = record;
        }
    }
}
