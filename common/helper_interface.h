#ifndef COMMON_HELPER_INTERFACE_H_
#define COMMON_HELPER_INTERFACE_H_

#pragma once

#include <helper_string.h>

#define HAS_ARG(name) checkCmdLineFlag(argc, (const char **)argv, name)
#define FLOAT_ARG(name) getCmdLineArgumentFloat(argc, (const char **)argv, name)
#define INT_ARG(name) getCmdLineArgumentInt(argc, (const char **)argv, name)
#define TABS "\t\t"

inline void print_help()
{
    fprintf(stderr,
        "Usage: tgjoin [options] file\n"
        "Options:\n"
        "  help" TABS "Display this information\n"
        "  j=<threshold>\tThreshold for the Jaccard similarity function\n"
        "  mem=<min>\tMinimum GPU memory to be left free, in MB. "
        "Assumed to be less than the required to join, defaults to 100\n");
}

inline void print_formaterr()
{
    fprintf(stderr, "Dataset file is malformed. "
                    "Expected size-ascending list of: [record-id][record-size][token-list]\n");
}

inline bool is_option(const char *arg)
{
    return *arg == '-'
        || strncasecmp(arg, "j=", 2) == 0
        || strncasecmp(arg, "mem=", 4) == 0
        || strcasecmp(arg, "h") == 0
        || strcasecmp(arg, "help") == 0;
}

#endif