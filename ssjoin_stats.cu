#include <ssjoin_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream,
                "Host to device" TABS "%dms\n"
                "Indexing" TABS "%dms\n"
                "Filtering" TABS "%dms\n"
                "Max indexed token\t%d\n"
                "Indexed entries" TABS "%d\n"
                "Overlap matrix size\t%ldMB\n"
                "Filter iterations\t%d\n"
                "Token queries" TABS "%d\n"
                "Index queries" TABS "%d\n",
                host2device_ms,
                indexing_ms,
                filtering_ms,
                token_map_limit - 1,
                indexed_entries,
                matrix_bytesize / 1000000,
                iterations,
                token_queries,
                index_queries);
    }
    else
    {
        fprintf(stream, "Error\n");
    }
}
