#include <helper_io.h>
#include <helper_mem.h>
#include <helper_util.h>
#include <ssjoin.h>

int main(int argc, char **argv)
{
    if (argc < 2 || HAS_ARG("h") || HAS_ARG("help"))
    {
        print_help();
        exit(EXIT_WAIVED);
    }

    for (int i = 1; i < argc; ++i)
    {
        if (is_option((const char *)argv[i]))
        {
            ++i;
            continue;
        }

        uint32_t *dataset;
        size_t size;
        if (load_dataset(argv[i], &dataset, &size) == 0)
        {
            dataset_stats stats;
            if (verify_dataset(dataset, size, stats) == 0) {
                stats.print(stderr);
                run_join(dataset, size, stats).print(stderr);
            }
            else { print_formaterr(); }
            SAFE_FREE(&dataset, &size);
        }
    }

    return EXIT_SUCCESS;
}
