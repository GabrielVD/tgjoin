#include <ssjoin_index.cuh>
#include <helper_cuda.h>
#include <similarity.cuh>

__global__ void count_tokens(
    const uint32_t *records_d,
    const int cardinality,
    uint32_t *count_d,
    const float threshold)
{
    const int stride = STRIDE();
    uint32_t token_max{0}, token_count{0};
    count_d += 3; // reserve 3 cells for [token_max, token_count, 0]

    for (int idx = IDX(); idx < cardinality; idx += stride)
    {
        auto start{records_d[idx]};
        const auto size{index_prefix_size(records_d[idx + 1] - start, threshold)};
        token_count += size;

        const auto end{start + size};
        do
        {
            const auto token{records_d[start]};
            token_max = token > token_max ? token : token_max;
            atomicAdd(count_d + token, 1);
        } while (++start < end);
    }
    atomicMax(count_d - 3, token_max);
    atomicAdd(count_d - 2, token_count);
}

__global__ void make_index(
    const uint32_t *records_d,
    const int cardinality,
    const uint32_t *token_map_d,
    const float threshold,
    uint32_t *count_d,
    index_record *inverted_index_d)
{
    const int stride = STRIDE();
    for (int idx = IDX(); idx < cardinality; idx += stride)
    {
        index_record record;
        record.start_index = records_d[idx];
        record.size = records_d[idx + 1] - record.start_index;
        const auto end{record.start_index
            + index_prefix_size(record.size, threshold)};
        
        for (auto pos{record.start_index}; pos < end; ++pos)
        {
            record.remaining_tokens = end - pos;
            const auto token{records_d[pos]};
            const auto head{token_map_d[token] + atomicSub(count_d + token, 1)};
            inverted_index_d[head] = record;
        }
    }
}

__global__ void sort_index(
    const uint32_t *token_map_d,
    const int token_map_size,
    index_record *inverted_index_d)
{
    const int stride = STRIDE();
    for (int token = IDX(); token < token_map_size - 1; token += stride)
    {
        const auto start{token_map_d[token]};
        const auto end{token_map_d[token + 1]};

        for (auto head{start + 1}; head < end; ++head)
        {
            const auto record{inverted_index_d[head]};
            auto i{head};
            for (; i > start
                && inverted_index_d[i-1].start_index < record.start_index; --i)
            {
                inverted_index_d[i] = inverted_index_d[i-1];
            }
            if (i != head) { inverted_index_d[i] = record; }
        }
        
    }
}
