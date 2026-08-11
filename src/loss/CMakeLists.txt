xlearn_add_library(loss
  SOURCES loss.cc squared_loss.cc cross_entropy_loss.cc metric.cc
  DEPS xlearn::score)

xlearn_add_test(loss_test xlearn::loss)

xlearn_add_test(squared_loss_test xlearn::loss)

xlearn_add_test(cross_entropy_loss_test xlearn::loss)

xlearn_add_test(metric_test xlearn::loss)

xlearn_add_benchmark(loss_bench xlearn::loss)
