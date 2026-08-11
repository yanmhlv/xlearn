# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/score)

# Build static library
set(STA_DEPS data base hwy)
add_library(score STATIC score_function.cc
linear_score.cc fm_score.cc ffm_score.cc)
target_link_libraries(score ${STA_DEPS})

# Build uinttests
set(LIBS score data base hwy)

xlearn_add_test(score_function_test ${LIBS})

xlearn_add_test(linear_score_test ${LIBS})

xlearn_add_test(fm_score_test ${LIBS})

xlearn_add_test(ffm_score_test ${LIBS})

# Build benchmarks.
xlearn_add_benchmark(score_bench ${LIBS})

# Install library and header files
install(TARGETS score DESTINATION lib/score)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/score)
