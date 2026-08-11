xlearn_add_library(solver
  SOURCES checker.cc trainer.cc inference.cc solver.cc
  DEPS xlearn::reader xlearn::loss)

if(WIN32)
  target_link_libraries(xlearn_solver PUBLIC Ws2_32)
endif()

# The command line tools sit at the top of the build tree, where the demo data
# and the example script are staged next to them.
add_executable(xlearn_train train_main.cc)
target_link_libraries(xlearn_train PRIVATE xlearn::solver)

add_executable(xlearn_predict predict_main.cc)
target_link_libraries(xlearn_predict PRIVATE xlearn::solver)

# The generator expression is what keeps a multi-configuration generator from
# appending the configuration to the path and landing the tools away from the
# data staged beside them.
set_target_properties(xlearn_train xlearn_predict PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<1:${PROJECT_BINARY_DIR}>")

if(NOT XLEARN_BUILD_PYTHON)
  install(TARGETS xlearn_train xlearn_predict)
endif()
