#ifndef COMMON_HELPER_UTIL_H_
#define COMMON_HELPER_UTIL_H_

#pragma once

#include <helper_string.h>

#define HAS_ARG(name) checkCmdLineFlag(argc, (const char **)argv, name)
#define STR_ARG(name, retval) getCmdLineArgumentString(argc, (const char **)argv, name, retval)
#define TABS "\t\t"

inline void print_help()
{
    fprintf(stderr, "help placeholder\n");
}

inline void print_formaterr()
{
    fprintf(stderr, "Dataset file is malformed. "
                    "Expected ascending list of: [record-id][record-size][token-list]\n");
}

inline bool is_option(const char *arg)
{
    return strcasecmp(arg, "h") == 0
        || strcasecmp(arg, "t") == 0
        || strcasecmp(arg, "help") == 0;
}

#endif