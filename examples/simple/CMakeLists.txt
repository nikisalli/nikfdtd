cmake_minimum_required (VERSION 3.0.0)
project (simple CXX)

add_executable(simple main.cpp)


target_link_libraries (simple fdtd)
