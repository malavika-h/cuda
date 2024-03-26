cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)

if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    message("Please use 64-bit cmake generator.")
endif()

project(CPP_Fluid_Particles VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Specify CUDA compiler
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")

# Suppress some warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Add your CUDA source files
file(GLOB CUDA_SOURCES src/*.cu src/*.h src/*.cpp)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
# Add executable target
cuda_add_executable(CPP_Fluid_Particles ${CUDA_SOURCES})
get_target_property(CPP_FLUID_PARTICLES_LINK_FLAGS CPP_Fluid_Particles LINK_FLAGS)

# Find GLEW and GLUT
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Include directories
include_directories(
    ${GLEW_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    /usr/local/cuda-12.3/cuda-samples/Common
)

# Link libraries
target_link_libraries(CPP_Fluid_Particles
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    cudart
    stdc++
)

# Adjusting the location where CUDA object files are generated
set_target_properties(CPP_Fluid_Particles PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_OBJECT_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/src")

