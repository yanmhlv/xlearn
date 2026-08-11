xlearn_add_library(base
  SOURCES logging.cc stringprintf.cc split_string.cc levenshtein_distance.cc
          timer.cc format_print.cc
  DEPS Threads::Threads)

xlearn_add_test(levenshtein_distance_test xlearn::base)

xlearn_add_test(file_util_test xlearn::base)

xlearn_add_test(thread_pool_test xlearn::base)
