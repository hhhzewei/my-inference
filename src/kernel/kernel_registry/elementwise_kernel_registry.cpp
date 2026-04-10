//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_key_generator/elementwise_key_generator.h"
#include "kernel/kernel_creator/generic_kernel_creator.h"
#include "kernel/kernel_creator/cpu/elementwise_with_stride_kernel_creator.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "util/math.h"

using namespace my_inference;
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Add,DataType::Float32,false),
    cpu::BinaryElementwiseKernel<float, AddFunctor<float>>);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Sub,DataType::Float32,false),
    cpu::BinaryElementwiseKernel<float, SubFunctor<float> >);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Mul,DataType::Float32,false),
    cpu::BinaryElementwiseKernel<float, MulFunctor<float> >);
GENERIC_REGISTER_KERNEL(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Div,DataType::Float32,false),
    cpu::BinaryElementwiseKernel<float, DivFunctor<float> >);
// with stride
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Add,DataType::Float32,true),
    &(cpu::BinaryElementwiseKernelWithStrideCreator<AddFunctor<float>,float>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Sub,DataType::Float32,true),
    &(cpu::BinaryElementwiseKernelWithStrideCreator<SubFunctor<float>,float>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Mul,DataType::Float32,true),
    &(cpu::BinaryElementwiseKernelWithStrideCreator<MulFunctor<float>,float>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementwiseKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Div,DataType::Float32,true),
    &(cpu::BinaryElementwiseKernelWithStrideCreator<DivFunctor<float>,float>::instance()));
