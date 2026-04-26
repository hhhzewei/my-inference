//
// Created by hzw on 2026/4/26.
//

#include "kernel/cpu/avx512/elementwise/unary_elementwise_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/clip_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    ClipKeyGenerator::generate(DeviceType::CPU,IsaType::Avx512,OpType::Clip,DataType::Float32,ClipType::
        Relu6),
    cpu::avx512::UnaryElementwiseKernel<float, UnaryOpType::Relu6>);
