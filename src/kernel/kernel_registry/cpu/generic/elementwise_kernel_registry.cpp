//
// Created by hzw on 2026/3/29.
//

#include "kernel/cpu/generic/elementwise/binary_elementwise_kernel.h"
#include "kernel/kernel_key_generator/elementwise_key_generator.h"
#include "kernel/kernel_creator/generic_kernel_creator.h"
#include "kernel/kernel_creator/cpu/elementwise_with_stride_kernel_creator.h"
#include "kernel/kernel_creator/kernel_creator_util.h"

using namespace my_inference;
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Add,DataType::Float32,false),
    cpu::generic::BinaryElementwiseKernel<float, BinaryOpType::Add>);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Sub,DataType::Float32,false),
    cpu::generic::BinaryElementwiseKernel<float, BinaryOpType::Sub>);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Mul,DataType::Float32,false),
    cpu::generic::BinaryElementwiseKernel<float, BinaryOpType::Mul>);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Div,DataType::Float32,false),
    cpu::generic::BinaryElementwiseKernel<float, BinaryOpType::Div>);
// broadcast
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Add,DataType::Float32,true),
    &(cpu::generic::BinaryElementwiseKernelWithStrideCreator<float,BinaryOpType::Add>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Sub,DataType::Float32,true),
    &(cpu::generic::BinaryElementwiseKernelWithStrideCreator<float,BinaryOpType::Sub>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Mul,DataType::Float32,true),
    &(cpu::generic::BinaryElementwiseKernelWithStrideCreator<float,BinaryOpType::Mul>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Div,DataType::Float32,true),
    &(cpu::generic::BinaryElementwiseKernelWithStrideCreator<float,BinaryOpType::Div>::instance()));
