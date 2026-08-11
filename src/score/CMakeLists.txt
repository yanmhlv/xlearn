xlearn_add_library(score
  SOURCES score_function.cc linear_score.cc fm_score.cc ffm_score.cc
  DEPS xlearn::data hwy)

xlearn_add_test(score_function_test xlearn::score)

xlearn_add_test(linear_score_test xlearn::score)

xlearn_add_test(fm_score_test xlearn::score)

xlearn_add_test(ffm_score_test xlearn::score)

xlearn_add_benchmark(score_bench xlearn::score)
