//
// Created by hzw on 2026/4/4.
//

#include "kernel/cpu/generic/clip_kernel.h"
#include "kernel/cpu/generic/elementwise_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/clip_key_generator.h"
#include "util/math.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    ClipKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Clip,DataType::Float32,ClipType::Standard),
    cpu::generic::ClipKernel<float>);
GENERIC_REGISTER_KERNEL(
    ClipKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Clip,DataType::Float32,ClipType::Relu6),
    cpu::generic::unaryElementwiseKernel<float, Relu6Functor<float>>);
