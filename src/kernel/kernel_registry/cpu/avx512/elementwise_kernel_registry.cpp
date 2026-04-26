//
// Created by hzw on 2026/4/26.
//

#include "kernel/cpu/avx512/elementwise/binary_elementwise_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/elementwise_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Avx512,OpType::Add,DataType::Float32,false),
    cpu::avx512::BinaryElementwiseKernel<float, BinaryOpType::Add>);
