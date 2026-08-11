xlearn_add_library(reader
  SOURCES parser.cc file_splitor.cc reader.cc
  DEPS xlearn::data)

xlearn_add_test(parser_test xlearn::reader)

xlearn_add_test(reader_test xlearn::reader)

xlearn_add_test(file_splitor_test xlearn::reader)

xlearn_add_benchmark(parser_bench xlearn::reader)
