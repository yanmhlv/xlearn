if(WIN32)
  set(runner run_example.bat)
else()
  set(runner run_example.sh)
endif()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/${runner}"
     DESTINATION "${PROJECT_BINARY_DIR}")
