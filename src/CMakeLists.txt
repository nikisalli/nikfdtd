cmake_minimum_required(VERSION 3.0.0)
project(fdtd LANGUAGES CXX)

enable_testing()

include_directories(${PROJECT_SOURCE_DIR})
add_executable(fdtd main.cpp draw.cpp kernel_H.comp kernel_E.comp)

find_package(SDL2 REQUIRED)
find_package(OpenMP)
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(fdtd ${SDL2_LIBRARIES})
target_link_libraries(fdtd OpenMP::OpenMP_CXX)
target_link_libraries(fdtd ${Vulkan_LIBRARIES})
target_link_libraries(fdtd ${OpenCV_LIBS})
target_include_directories(fdtd PUBLIC ${Vulkan_INCLUDE_DIR})

function(add_spirv_shader INPUT_FILE OUTPUT_FILE)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
            COMMAND glslc ${INPUT_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
            MAIN_DEPENDENCY ${INPUT_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()


add_spirv_shader(${CMAKE_SOURCE_DIR}/src/kernel_H.comp H.spv)
add_spirv_shader(${CMAKE_SOURCE_DIR}/src/kernel_E.comp E.spv)
