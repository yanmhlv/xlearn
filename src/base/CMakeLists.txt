# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/base)

# Build static library
add_library(base STATIC logging.cc stringprintf.cc split_string.cc 
levenshtein_distance.cc timer.cc format_print.cc)

# Build unittests.
if(NOT WIN32)
set(LIBS base pthread)
else(WIN32)
set(LIBS base)
endif()

xlearn_add_test(levenshtein_distance_test ${LIBS})

xlearn_add_test(file_util_test ${LIBS})

xlearn_add_test(thread_pool_test ${LIBS})

# Install library and header files
install(TARGETS base DESTINATION lib/base)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/base)
