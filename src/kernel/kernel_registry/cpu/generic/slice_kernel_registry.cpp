//
// Created by hzw on 2026/4/24.
//

#include "kernel/cpu/generic/slice_kernel.h"
#include "kernel/kernel_creator/kernel_creator_util.h"
#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"

using namespace my_inference;
REGISTER_KERNEL_CREATOR(
    GenericKernelKeyGenerator::generate(DeviceType::CPU,IsaType::Generic,OpType::Slice,DataType::Float32),
    &GenericKernelCreator<cpu::generic::SliceKernel<float>>::instance());
