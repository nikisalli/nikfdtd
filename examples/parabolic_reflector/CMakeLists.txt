cmake_minimum_required (VERSION 3.0.0)
project (parabolic_reflector CXX)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_executable(parabolic_reflector main.cpp)

target_link_libraries (parabolic_reflector fdtd)
