//
// Created by hzw on 2026/4/4.
//
#include "kernel/cpu/conv/standard_conv2D_kernel.h"
#include "kernel/cpu/conv/depthwise_conv2D_kernel.h"
#include "kernel/cpu/conv/grouped_conv2D_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/conv_key_generator.h"

using namespace my_inference;
GENERIC_REGISTER_KERNEL(
    cpu::ConvKeyGenerator::generate(DeviceType::CPU,isa_type::Default,OpType::Conv,DataType::Float32,2,ConvType::Standard),
    cpu::StandardConv2DKernel<float>);
GENERIC_REGISTER_KERNEL(
    cpu::ConvKeyGenerator::generate(DeviceType::CPU,isa_type::Default,OpType::Conv,DataType::Float32,2,ConvType::Depthwise),
    cpu::DepthwiseConv2dKernel<float>);
GENERIC_REGISTER_KERNEL(
    cpu::ConvKeyGenerator::generate(DeviceType::CPU,isa_type::Default,OpType::Conv,DataType::Float32,2,ConvType::Grouped),
    cpu::GroupedConv2DKernel<float>);
