# Every demo is staged under the build tree as one directory holding its data,
# its Python scripts and the runner script of the platform being built for.
function(xlearn_stage_demo path)
  if(WIN32)
    set(runner "*.bat")
  else()
    set(runner "*.sh")
  endif()

  file(GLOB files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${path}/*.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/${path}/*.csv"
    "${CMAKE_CURRENT_SOURCE_DIR}/${path}/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/${path}/${runner}")

  get_filename_component(name "${path}" NAME)
  file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/demo/${name}")
endfunction()

xlearn_stage_demo(classification/criteo_ctr)
xlearn_stage_demo(classification/higgs)
xlearn_stage_demo(classification/mushroom)
xlearn_stage_demo(classification/titanic)
xlearn_stage_demo(classification/scikit_learn_demo)
xlearn_stage_demo(regression/house_price)

# The example script that scripts/ stages next to the command line tools reads
# the criteo sample from beside them.
file(COPY
  "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_train.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_test.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.py"
  DESTINATION "${PROJECT_BINARY_DIR}")
