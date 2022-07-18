#include <ssjoin_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream,
                "Host to device" TABS "%.3lfms\n"
                "Indexing" TABS "%.3lfms\n"
                "Filtering" TABS "%.3lfms\n"
                "Total Time" TABS "%.3lfms\n"
                "Max indexed token\t%d\n"
                "Indexed entries" TABS "%d\n"
                "Overlap matrix size\t%ldMB\n"
                "Filter iterations\t%d\n"
                "Token probes" TABS "%d\n"
                "Index probes" TABS "%d\n",
                host2device_ms,
                indexing_ms,
                filtering_ms,
                host2device_ms + indexing_ms + filtering_ms,
                token_map_limit - 1,
                indexed_entries,
                matrix_bytesize / 1000000,
                iterations,
                token_probes,
                index_probes);
    }
    else
    {
        fprintf(stream, "Error\n");
    }
}
