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
    const record_t* __restrict__ record_map_d,
    record_t key_start,
    const record_t key_limit,
    const record_t* __restrict__ token_map_d,
    const record_t token_limit,
    const index_record* __restrict__ index_d,
    const float threshold,
    const float overlap_factor,
    overlap_t* __restrict__ overlap_matrix_d,
    const size_t overlap_offset)
{
    const record_t stride = STRIDE();
    for (key_start += IDX(); key_start < key_limit; key_start += stride)
    {
        auto *overlap_row{overlap_matrix_d + (tri_rowstart(key_start) - overlap_offset)};
        auto head{record_map_d[key_start]};
        const auto record_end{record_map_d[key_start + 1] - 2};
        const auto size{record_end - head};
        const auto minsize{MINSIZE_D(size, threshold)};
        const auto prefix_end{head + prefix_size_d(size, threshold)};
        
        while (head < prefix_end)
        {
            const auto token{record_map_d[head]};
            ++head;
            if (token < token_limit)
            {
                auto candidate{token_map_d[token]};
                const auto list_end{token_map_d[token + 1]};
                
                for (; candidate < list_end; ++candidate)
                {
                    const auto record{index_d[candidate]};
                    record_t current_overlap;
                    if (key_start > record.key
                    && record.size >= minsize
                    && (current_overlap = overlap_row[record.key]) < LIMIT - 1)
                    {
                        ++current_overlap;
                        const auto overlap{OVERLAP_D(size, record.size, overlap_factor)};
                        const auto max_overlap{current_overlap
                            + min_d(record.remaining_tokens, record_end - head)};
                        current_overlap = max_overlap >= overlap ? current_overlap : LIMIT;
                        overlap_row[record.key] = current_overlap;
                    }
                }
            }
        }
    }
}
