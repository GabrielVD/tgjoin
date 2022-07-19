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
    const record_t *records_d,
    record_t id_start,
    const record_t id_limit,
    const record_t *token_map_d,
    const record_t token_limit,
    const index_record *inverted_index_d,
    const float threshold,
    const float overlap_factor,
    uint8_t *matrix_tip_d,
    record_t *stats_d) // [token_probes, index_probes]
{
    record_t token_probes{0}, index_probes{0};
    const record_t stride = STRIDE();
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
            token_probes += prefix;
        }
        while (start < end)
        {
            const auto token{records_d[start]};
            ++start;
            if (token < token_limit)
            {
                auto candidate{token_map_d[token]};
                const auto list_end{token_map_d[token + 1]};
                index_probes += list_end - candidate;
                for (; candidate < list_end; ++candidate)
                {
                    const auto record{inverted_index_d[candidate]};
                    record_t current_overlap;
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
    atomicAdd(stats_d, token_probes);
    atomicAdd(stats_d + 1, index_probes);
}
