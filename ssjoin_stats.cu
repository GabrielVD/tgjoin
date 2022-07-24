#include <ssjoin_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    switch (status)
    {
    case ssjoin_status::SUCCESS:
        fprintf(stream,
                "File read" TABS "%.3lfms\n"
                "Host to device" TABS "%.3lfms\n"
                "Indexing" TABS "%.3lfms\n"
                "Filtering" TABS "%.3lfms\n"
                "Total join time" TABS "%.3lfms\n"
                "GPU memory pool" TABS "%ldMB\n"
                "Max indexed token\t%d\n"
                "Indexed entries" TABS "%d\n"
                "Filter iterations\t%d\n"
                "Token probes" TABS "%d\n"
                "Index probes" TABS "%d\n",
                read_ms,
                host2device_ms,
                indexing_ms,
                filtering_ms,
                host2device_ms + indexing_ms + filtering_ms,
                pool_size / 1000000,
                token_map_limit - 1,
                indexed_entries,
                iterations,
                token_probes,
                index_probes);
        break;

    case ssjoin_status::IO_ERR:
        fprintf(stream, "Error during file operation\n");
        break;

    case ssjoin_status::FORMAT_ERR:
        fprintf(stderr, "Dataset file is malformed. "
                        "Expected size-ascending list of: [record-id][record-size][token-list]\n");
        break;
        
    default:
        fprintf(stream, "Error\n");
    }
}
