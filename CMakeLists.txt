cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(pdb-sph)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH /usr/local/lib/cmake)
set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/glfw)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.c ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cu ${CMAKE_SOURCE_DIR}/src/*.cuda)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.hpp)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA 10 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(pdb-sph ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCE_FILES})


include_directories("${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include" "/usr/local/include" "${CMAKE_SOURCE_DIR}/third_party" "/usr/include/GL" "/usr/local/cuda/include")

target_link_libraries(pdb-sph ${OPENGL_LIBRARIES} glfw ${GLEW_LIBRARIES})



