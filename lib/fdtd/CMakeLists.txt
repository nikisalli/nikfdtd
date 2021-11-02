cmake_minimum_required(VERSION 3.0.0)
project(fdtd LANGUAGES CXX)

set (FDTD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (FDTD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(FDTD_SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fdtd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_E.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_H.comp
)

include_directories("${FDTD_SOURCE_DIR}")
include_directories("${FDTD_INCLUDE_DIR}")

add_library(fdtd SHARED ${FDTD_SRCS} ${FDTD_INCLUDE_DIR})

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

target_include_directories (fdtd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)