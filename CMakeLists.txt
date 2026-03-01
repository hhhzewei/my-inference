cmake_minimum_required(VERSION 4.0)

set(CMAKE_TOOLCHAIN_FILE "D:/CppDevelop/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(my_inference CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    # 强制 MSVC 使用 UTF-8 编码读取和生成字符
    add_compile_options(/utf-8)
endif ()

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(protobuf CONFIG REQUIRED)
find_package(onnx CONFIG REQUIRED)

add_library(my_inference STATIC src/graph/graph.cpp
        src/graph/attribute_propagate/conv_attr_propagator.cpp
        src/graph/attribute_propagate/gemm_attr_propagator.cpp
        src/graph/date_type_infer/data_type_infer.cpp
        src/graph/shape_infer/identity_shape_infer.cpp
        src/graph/shape_infer/broadcast_shape_infer.cpp
        src/graph/shape_infer/conv_shape_infer.cpp
        src/graph/shape_infer/gemm_shape_infer.cpp
        src/graph/shape_infer/stride.cpp
        src/optimize/dead_code_elimination.cpp
        src/optimize/constant_folding.cpp
        src/util/util.cpp
        src/util/onnx_util.cpp
)

target_link_libraries(my_inference
        PUBLIC ONNX::onnx ONNX::onnx_proto
        PUBLIC protobuf::libupb protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

add_subdirectory(test/MobileNet-v2)
add_subdirectory(test/optimize)
