cmake_minimum_required (VERSION 3.0.0)
project (optical_ring_resonator CXX)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_compile_options(-Wno-narrowing)

add_executable(optical_ring_resonator main.cpp)

target_link_libraries (optical_ring_resonator fdtd)
