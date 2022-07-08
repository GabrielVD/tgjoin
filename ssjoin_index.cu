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
    count_d += 2; // reserve 2 cells for [token_max, token_count]

    for (int idx = IDX(); idx < cardinality; idx += stride)
    {
        auto start{records_d[idx]};
        auto size{records_d[idx + 1] - start};
        size = size + 1 - OVERLAP(threshold, size, size);
        token_count += size;

        const auto end{start + size};
        do
        {
            auto token{records_d[start]};
            token_max = token > token_max ? token : token_max;
            atomicAdd(count_d + token, 1);
        } while (++start < end);
    }
    atomicMax(count_d - 2, token_max);
    atomicAdd(count_d - 1, token_count);
}
