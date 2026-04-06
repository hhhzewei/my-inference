//
// Created by hzw on 2026/4/6.
//
#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"

using namespace my_inference;
REGISTER_KERNEL_KEY_GENERATOR(OpType::ReduceMax, &GenericKernelKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::ReduceMin, &GenericKernelKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::ReduceMean, &GenericKernelKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::ReduceSum, &GenericKernelKeyGenerator::instance());
