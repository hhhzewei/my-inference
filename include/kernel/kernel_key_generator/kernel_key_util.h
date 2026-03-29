//
// Created by hzw on 2026/3/29.
//
#ifndef MY_INFERENCE_KERNEL_KEY_UTIL_H
#define MY_INFERENCE_KERNEL_KEY_UTIL_H

#include "graph/node/op_node.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/factory.h"

#define REGISTER_KERNEL_KEY_GENERATOR(op_type,kernel_key_generator) GENERIC_REGISTER(my_inference::OpType,my_inference::KernelKeyGenerator *,op_type,kernel_key_generator);

my_inference::KernelKey getKernelKey(const my_inference::OpNode *op);

#endif //MY_INFERENCE_KERNEL_KEY_UTIL_H
