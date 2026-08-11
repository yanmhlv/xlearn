# nanobind ships its CMake config inside the Python package, which the build
# backend installs into the isolated build environment.
find_package(Python 3.12 REQUIRED
  COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

file(READ "${PROJECT_SOURCE_DIR}/python-package/xlearn/VERSION" XLEARN_VERSION)
string(STRIP "${XLEARN_VERSION}" XLEARN_VERSION)

nanobind_add_module(_core STABLE_ABI NB_STATIC bind.cc)
target_compile_definitions(_core PRIVATE XLEARN_VERSION="${XLEARN_VERSION}")

if(NOT WIN32)
set(LIBS solver reader loss score data base hwy pthread)
else(WIN32)
set(LIBS solver reader loss score data base hwy Ws2_32)
endif()
target_link_libraries(_core PRIVATE ${LIBS})

install(TARGETS _core LIBRARY DESTINATION xlearn)

# The signatures nanobind knows are only in the compiled module, so the stub
# is generated from it rather than written by hand.
nanobind_add_stub(_core_stub
  MODULE _core
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi" DESTINATION xlearn)
