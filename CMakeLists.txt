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
        src/graph/node/tensor_node.cpp
        src/graph/node/op_node.cpp
        # attribute propagate
        src/graph/attribute_propagate/attr_propagate_util.cpp
        src/graph/attribute_propagate/conv_attr_propagator.cpp
        src/graph/attribute_propagate/gemm_attr_propagator.cpp
        src/graph/attribute_propagate/batch_norm_attr_propagator.cpp
        src/graph/attribute_propagate/reduce_attr_propagator.cpp
        # data type inference
        src/graph/date_type_infer/data_type_infer.cpp
        # shape inference
        src/graph/shape_infer/shape_infer_util.cpp
        src/graph/shape_infer/identity_shape_infer.cpp
        src/graph/shape_infer/broadcast_shape_infer.cpp
        src/graph/shape_infer/conv_shape_infer.cpp
        src/graph/shape_infer/gemm_shape_infer.cpp
        src/graph/shape_infer/reduce_shape_infer.cpp
        src/graph/shape_infer/stride.cpp
        # dce
        src/optimize/dead_code_elimination.cpp
        # constant folding
        src/optimize/constant_folding.cpp
        src/optimize/constant_folder/op_folder_util.cpp
        src/optimize/constant_folder/elementwise_folder.cpp
        # cse
        src/optimize/common_subexpression_elimination.cpp
        # op fusion
        src/optimize/op_fusion.cpp
        src/optimize/op_fusion/op_fusion_pattern.cpp
        src/optimize/op_fusion/conv_batchnorm_fuser.cpp
        # memory planning
        src/memory/memory_planning.cpp
        # kernel key
        src/kernel/kernel_key_generator/kernel_key_util.cpp
        src/kernel/kernel_key_generator/elementwise_key_generator.cpp
        src/kernel/kernel_key_generator/batch_norm_key_generator.cpp
        src/kernel/kernel_key_generator/conv_key_generator.cpp
        src/kernel/kernel_key_generator/clip_key_generator.cpp
        src/kernel/kernel_key_generator/gemm_key_generator.cpp
        src/kernel/kernel_key_generator/reduce_key_generator.cpp
        #kernel registry
        src/kernel/kernel_util.cpp
        src/kernel/kernel_registry/elementwise_kernel_registry.cpp
        src/kernel/kernel_registry/batch_norm_registry.cpp
        src/kernel/kernel_registry/con2D_kernel_registry.cpp
        src/kernel/kernel_registry/clip_kernel_registry.cpp
        src/kernel/kernel_registry/reduce_kernel_registry.cpp
        src/kernel/kernel_registry/gemm_kernel_registry.cpp
        #memory allocator
        src/memory/memory_allocator/memory_allocator_util.cpp
        src/memory/memory_allocator/cpu_memory_allocator.cpp
        # util
        src/util/util.cpp
        src/util/onnx_util.cpp
)

# 静态库中的注册声明对于main函数是死代码，会被链接器移除，必须用以下链接选项避免注册失效
target_link_options(my_inference INTERFACE
        "$<DEVICE_LINK:/WHOLEARCHIVE:$<TARGET_FILE:my_inference>>"
        "$<HOST_LINK:/WHOLEARCHIVE:$<TARGET_FILE:my_inference>>"
)

target_link_libraries(my_inference
        PUBLIC ONNX::onnx ONNX::onnx_proto
        PUBLIC protobuf::libupb protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

add_subdirectory(test/MobileNet-v2)
add_subdirectory(test/optimize)
