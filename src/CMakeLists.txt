cmake_minimum_required(VERSION 3.0.0)
project(fdtd LANGUAGES CXX)

enable_testing()

include_directories(${PROJECT_SOURCE_DIR})
add_executable(fdtd main.cpp draw.cpp)

find_package(SDL2 REQUIRED)
find_package(OpenMP)
find_package(Vulkan REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
target_link_libraries(fdtd ${SDL2_LIBRARIES})
target_link_libraries(fdtd OpenMP::OpenMP_CXX)
target_link_libraries(fdtd ${Vulkan_LIBRARIES})
target_include_directories(fdtd PUBLIC ${Vulkan_INCLUDE_DIR})