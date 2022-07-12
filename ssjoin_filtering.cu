#include <helper_cuda.h>
#include <helper_mem.cuh>
#include <similarity.cuh>
#include <limits>
#include <algorithm>

#define LIMIT ((uint8_t)-1)

template<typename T>
__device__ static T min_d(T a, T b)
{
    return a < b ? a : b;
}

__global__ void filter(
    const uint32_t *records_d,
    int id_start,
    const int id_limit,
    const uint32_t *token_map_d,
    const uint32_t token_limit,
    const index_record *inverted_index_d,
    const float threshold,
    const float overlap_factor,
    uint8_t *matrix_tip_d,
    uint32_t *stats_d) // [token_queries, index_queries]
{
    uint32_t token_queries{0}, index_queries{0};
    const int stride = STRIDE();
    for (id_start += IDX(); id_start < id_limit; id_start += stride)
    {
        auto *overlap_row{matrix_tip_d + tri_rowstart(id_start)};
        auto start{records_d[id_start]};
        const auto record_end{records_d[id_start + 1]};
        const auto size{record_end - start};
        const auto minsize{MINSIZE(size, threshold)};
        auto end{start};
        {
            const auto prefix{prefix_size(size, threshold)};
            end += prefix;
            token_queries += prefix;
        }
        while (start < end)
        {
            const auto token{records_d[start]};
            ++start;
            if (token < token_limit)
            {
                auto candidate{token_map_d[token]};
                const auto list_end{token_map_d[token + 1]};
                index_queries += list_end - candidate;
                for (; candidate < list_end; ++candidate)
                {
                    const auto record{inverted_index_d[candidate]};
                    uint32_t current_overlap;
                    if (id_start > record.id
                    && record.size >= minsize
                    && (current_overlap = overlap_row[record.id]) < LIMIT - 1)
                    {
                        ++current_overlap;
                        const auto overlap{OVERLAP(size, record.size, overlap_factor)};
                        const auto max_overlap{current_overlap
                            + min_d(record.remaining_tokens, record_end - start)};
                        current_overlap = max_overlap >= overlap ? current_overlap : LIMIT;
                        overlap_row[record.id] = current_overlap;
                    }
                }
            }
        }
    }
    atomicAdd(stats_d, token_queries);
    atomicAdd(stats_d + 1, index_queries);
}
