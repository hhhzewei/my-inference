//
// Created by hzw on 2026/4/4.
//

#include "kernel/cpu/avx512/elementwise/unary_elementwise_kernel.h"
#include "kernel/cpu/generic/clip_kernel.h"
#include "kernel/cpu/generic/elementwise/unary_elementwise_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/clip_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    ClipKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Clip,DataType::Float32,ClipType::Standard),
    cpu::generic::ClipKernel<float>);
GENERIC_REGISTER_KERNEL(
    ClipKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Clip,DataType::Float32,ClipType::Relu6),
    cpu::generic::UnaryElementwiseKernel<float, UnaryOpType::Relu6>);
