//
// Created by hzw on 2026/4/5.
//

#include "kernel/cpu/reduce_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(GenericKernelKeyGenerator::generate(OpType::ReduceMax,DeviceType::CPU,DataType::Float32),
                        cpu::ReduceMaxKernel<float>);
GENERIC_REGISTER_KERNEL(GenericKernelKeyGenerator::generate(OpType::ReduceMin,DeviceType::CPU,DataType::Float32),
                        cpu::ReduceMinKernel<float>);
GENERIC_REGISTER_KERNEL(GenericKernelKeyGenerator::generate(OpType::ReduceMean,DeviceType::CPU,DataType::Float32),
                        cpu::ReduceMeanKernel<float>);
GENERIC_REGISTER_KERNEL(GenericKernelKeyGenerator::generate(OpType::ReduceSum,DeviceType::CPU,DataType::Float32),
                        cpu::ReduceSumKernel<float>);
