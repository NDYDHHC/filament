cmake_minimum_required(VERSION 3.19)

# ==================================================================================================
# Library
# ==================================================================================================
project(shader-transform-sandbox-lang)
set(TARGET shader-transform-sandbox-lang)

file(GLOB_RECURSE HDRS src/shader-transform-sandbox/*.h)
file(GLOB_RECURSE SRCS src/shader-transform-sandbox/*.cpp)

message(STATUS "HDRS: '$HDRS'")
message(STATUS "SRCS: '$SRCS'")

# ==================================================================================================
# Sources and headers
# ==================================================================================================

# ==================================================================================================
# Target definitions
# ==================================================================================================
add_library(${TARGET} STATIC ${SRCS} ${HDRS})
target_include_directories(${TARGET} PUBLIC src ${dawn_SOURCE_DIR}/third_party/tint/include)
set_target_properties(${TARGET} PROPERTIES FOLDER Tools)

target_link_libraries(${TARGET} glslang SPIRV SPIRV-Tools-opt spirv-cross-glsl)

# =================================================================================================
# Licenses
# ==================================================================================================
set(MODULE_LICENSES getopt glslang spirv-cross spirv-tools smol-v)
set(GENERATION_ROOT ${CMAKE_CURRENT_BINARY_DIR}/generated)
list_licenses(${GENERATION_ROOT}/licenses/licenses.inc ${MODULE_LICENSES})
target_include_directories(${TARGET} PRIVATE ${GENERATION_ROOT})

# ==================================================================================================
# Binary
# ==================================================================================================
project(shader-transform-sandbox)
set(TARGET shader-transform-sandbox)

# ==================================================================================================
# Sources and headers
# ==================================================================================================
set(SRCS src/main.cpp)

# ==================================================================================================
# Target definitions
# ==================================================================================================
add_executable(${TARGET} ${SRCS} ${HDRS})
target_link_libraries(${TARGET} shader-transform-sandbox-lang)
set_target_properties(${TARGET} PROPERTIES FOLDER Tools)

# ==================================================================================================
# Installation
# ==================================================================================================
install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
