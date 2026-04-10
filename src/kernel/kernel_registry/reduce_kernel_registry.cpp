//
// Created by hzw on 2026/4/5.
//

#include "kernel/cpu/generic/reduce_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    GenericKernelKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::ReduceMax,DataType::Float32),
    cpu::generic::ReduceMaxKernel<float>);
GENERIC_REGISTER_KERNEL(
    GenericKernelKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::ReduceMin,DataType::Float32),
    cpu::generic::ReduceMinKernel<float>);
GENERIC_REGISTER_KERNEL(
    GenericKernelKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::ReduceMean,DataType::Float32),
    cpu::generic::ReduceMeanKernel<float>);
GENERIC_REGISTER_KERNEL(
    GenericKernelKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::ReduceSum,DataType::Float32),
    cpu::generic::ReduceSumKernel<float>);
