#include <helper_cuda.h>
#include <similarity.cuh>

__global__ void count_tokens(
    const uint32_t *records_d,
    const int cardinality,
    uint32_t *count_d,
    const float threshold)
{
    const int stride = STRIDE();
    for (int idx = IDX(); idx < cardinality; idx += stride)
    {
        auto start{records_d[idx]};
        auto size{records_d[idx + 1] - start};
        size = size + 1 - OVERLAP(threshold, size, size);

        const auto end{start + size};
        do
        {
            auto token{records_d[start]};
            atomicAdd(count_d + token, 1);
        } while (++start < end);
    }
}
