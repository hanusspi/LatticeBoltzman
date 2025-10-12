cmake_minimum_required(VERSION 3.18)
project(LBM_LidDrivenCavity LANGUAGES CXX CUDA)

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

# Project sources
set(PROJECT_SOURCES
    source/main.cpp
)

# Project CUDA sources
set(PROJECT_CUDA_SOURCES
    source/lbm_kernel.cu
)

# Create executable
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_CUDA_SOURCES}
    ${GLAD_SOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/glad/include
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/glm
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glm
)

# Find OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Platform-specific settings
if(WIN32)
    # Windows-specific flags
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Debug/Release flags
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>
)

file(COPY ${CMAKE_SOURCE_DIR}/shaders
     DESTINATION ${CMAKE_BINARY_DIR})
