#include <runtime_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream,
                "Host to device" TABS "%dms\n"
                "Indexing" TABS "%dms\n"
                "Max indexed token\t%d\n"
                "Indexed entries" TABS "%d\n",
                host2device_ms,
                indexing_ms,
                token_map_size - 1,
                indexed_entries);
    }
    else
    {
        fprintf(stream, "Error\n");
    }
}
