cmake_minimum_required (VERSION 3.0.0)
project (simple CXX)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_executable(simple main.cpp)

target_link_libraries (simple fdtd)
