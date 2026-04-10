//
// Created by hzw on 2026/4/4.
//
#include "kernel/cpu/generic/conv/standard_conv2D_kernel.h"
#include "kernel/cpu/generic/conv/depthwise_conv2D_kernel.h"
#include "kernel/cpu/generic/conv/grouped_conv2D_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/conv_key_generator.h"

using namespace my_inference;
GENERIC_REGISTER_KERNEL(
    ConvKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Conv,DataType::Float32,2,ConvType::Standard),
    cpu::generic::StandardConv2DKernel<float>);
GENERIC_REGISTER_KERNEL(
    ConvKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Conv,DataType::Float32,2,ConvType::Depthwise),
    cpu::generic::DepthwiseConv2dKernel<float>);
GENERIC_REGISTER_KERNEL(
    ConvKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Conv,DataType::Float32,2,ConvType::Grouped),
    cpu::generic::GroupedConv2DKernel<float>);
