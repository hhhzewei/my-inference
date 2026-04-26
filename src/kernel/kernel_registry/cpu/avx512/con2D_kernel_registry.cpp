//
// Created by hzw on 2026/4/26.
//

#include "kernel/cpu/avx512/conv/depthwise_conv2D_kernel.h"
#include "kernel/cpu/avx512/conv/standard_conv2D_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/conv_key_generator.h"

using namespace my_inference;

GENERIC_REGISTER_KERNEL(
    ConvKeyGenerator::generate(DeviceType::CPU,IsaType::Avx512,OpType::Conv,DataType::Float32,2,ConvType::Standard),
    cpu::avx512::StandardConv2DKernel<float>);
GENERIC_REGISTER_KERNEL(
    ConvKeyGenerator::generate(DeviceType::CPU,IsaType::Avx512,OpType::Conv,DataType::Float32,2,ConvType::Depthwise),
    cpu::avx512::DepthwiseConv2DKernel<float>);
