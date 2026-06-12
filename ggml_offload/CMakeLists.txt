# ggml offload runtime for vits-melo-tts-zh_en-8k (flow + dec on ggml;
# CUDA on the Jetson Nano gen1 via the sensevoice-proven toolchain).
#
# Standalone build (GGML_DIR must point at a ggml source tree):
#   cmake -B build -DGGML_DIR=/path/to/ggml [-DGGML_CUDA=ON ...]
#   cmake --build build -j
#
# Inside the iPhO firmware tree the default GGML_DIR resolves to the
# sensevoice-vendored ggml (the CUDA-10.2/sm_53-proven copy).
cmake_minimum_required(VERSION 3.16)
project(melo8k_offload C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../sensevoice_cpp/sense-voice/csrc/third-party/ggml"
    CACHE PATH "ggml source tree")
if(NOT TARGET ggml)
  add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endif()

add_executable(melo8k-parity src/melo8k.cpp)
target_link_libraries(melo8k-parity PRIVATE ggml)
target_include_directories(melo8k-parity PRIVATE ${GGML_DIR}/include)
