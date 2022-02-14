cmake_minimum_required (VERSION 3.0.0)
project (fresnel CXX)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_compile_options(-Wno-narrowing)

add_executable(fresnel main.cpp)

target_link_libraries (fresnel fdtd)
