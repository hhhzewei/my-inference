//
// Created by hzw on 2026/4/1.
//
#include "kernel/cpu/batch_norm_kernel.h"
#include "kernel/kernel_creator/generic_kernel_creator.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"

using namespace my_inference;
REGISTER_KERNEL_CREATOR(
    my_inference::GenericKernelKeyGenerator::generate(DeviceType::CPU,isa_type::Default,OpType::BatchNormalization,
        DataType::Float32),
    &GenericKernelCreator<cpu::BatchNormKernel<float>>::instance());
