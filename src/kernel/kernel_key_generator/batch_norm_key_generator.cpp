//
// Created by hzw on 2026/4/1.
//


#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"
using namespace my_inference;

REGISTER_KERNEL_KEY_GENERATOR(OpType::BatchNormalization, &GenericKernelKeyGenerator::instance());
