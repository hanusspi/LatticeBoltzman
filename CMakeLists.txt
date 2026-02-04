cmake_minimum_required(VERSION 3.18)
project(LBM_Simulations LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Add external libraries
add_subdirectory(extern/glfw)
add_subdirectory(extern/glm)

# GLAD sources
set(GLAD_SOURCES
    extern/glad/src/glad.c
)

# ============================================================================
# Shared CUDA Library
# ============================================================================
add_library(lbm_core STATIC
    source/lbm_kernel.cu
)

target_include_directories(lbm_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/glad/include
)

set_target_properties(lbm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA fast math for performance
target_compile_options(lbm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ============================================================================
# Executable 1: Lid-Driven Cavity
# ============================================================================
add_executable(LBM_LidCavity
    source/main_lid_cavity.cpp
    ${GLAD_SOURCES}
)

target_include_directories(LBM_LidCavity PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/glad/include
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/glm
)

target_link_libraries(LBM_LidCavity PRIVATE
    lbm_core
    glfw
    glm
)

# Find and link OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(LBM_LidCavity PRIVATE OpenGL::GL)

set_target_properties(LBM_LidCavity PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# Executable 2: Flow Over Obstacle
# ============================================================================
add_executable(LBM_FlowObstacle
    source/main_flow_obstacle.cpp
    ${GLAD_SOURCES}
)

target_include_directories(LBM_FlowObstacle PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/glad/include
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/glm
)

target_link_libraries(LBM_FlowObstacle PRIVATE
    lbm_core
    glfw
    glm
    OpenGL::GL
)

set_target_properties(LBM_FlowObstacle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# Build Configuration
# ============================================================================

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Debug flags
target_compile_options(lbm_core PRIVATE
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>
)

# Copy shaders to build directory
file(COPY ${CMAKE_SOURCE_DIR}/shaders
     DESTINATION ${CMAKE_BINARY_DIR})
