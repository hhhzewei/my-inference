//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_key_generator/element_wise_kernel_key_generator.h"
#include "kernel/kernel_creator/generic_kernel_creator.h"
#include "kernel/kernel_creator/element_wise_with_stride_kernel_creator.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "util/math.h"

using namespace my_inference;
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Add,DeviceType::CPU,DataType::Float32,false),
    &(GenericKernelCreator<my_inference::cpu::BinaryElementWiseKernel<float, AddFunctor<float> > >::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Sub,DeviceType::CPU,DataType::Float32,false),
    &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, SubFunctor<float> > >::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Mul,DeviceType::CPU,DataType::Float32,false),
    &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, MulFunctor<float> > >::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Div,DeviceType::CPU,DataType::Float32,false),
    &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, DivFunctor<float> > >::instance()));
// with stride
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Add,DeviceType::CPU,DataType::Float32,true),
    &(BinaryElementWiseKernelWithStrideCreator<AddFunctor<float>,float,DeviceType::CPU>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Sub,DeviceType::CPU,DataType::Float32,true),
    &(BinaryElementWiseKernelWithStrideCreator<SubFunctor<float>,float,DeviceType::CPU>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Mul,DeviceType::CPU,DataType::Float32,true),
    &(BinaryElementWiseKernelWithStrideCreator<MulFunctor<float>,float,DeviceType::CPU>::instance()));
REGISTER_KERNEL_CREATOR(
    ElementWiseKeyGenerator::generate(OpType::Div,DeviceType::CPU,DataType::Float32,true),
    &(BinaryElementWiseKernelWithStrideCreator<DivFunctor<float>,float,DeviceType::CPU>::instance()));
