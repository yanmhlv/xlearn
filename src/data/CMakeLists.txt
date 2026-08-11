xlearn_add_library(data
  SOURCES model_parameters.cc
  DEPS xlearn::base)

xlearn_add_test(data_structure_test xlearn::data)

xlearn_add_test(model_parameters_test xlearn::data)
