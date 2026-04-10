//
// Created by hzw on 2026/4/5.
//

#include "kernel/cpu/generic/gemm_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/gemm_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    GemmKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Gemm,DataType::Float32,false,false),
    cpu::generic::GemmKernel<float, false, false>);
GENERIC_REGISTER_KERNEL(
    GemmKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Gemm,DataType::Float32,false,true),
    cpu::generic::GemmKernel<float, false, true>);
GENERIC_REGISTER_KERNEL(
    GemmKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Gemm,DataType::Float32,true,false),
    cpu::generic::GemmKernel<float, true, false>);
GENERIC_REGISTER_KERNEL(
    GemmKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Gemm,DataType::Float32,true,true),
    cpu::generic::GemmKernel<float, true, true>);
