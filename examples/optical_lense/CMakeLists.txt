cmake_minimum_required (VERSION 3.0.0)
project (optical_lense CXX)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_executable(optical_lense main.cpp)

target_link_libraries (optical_lense fdtd)
