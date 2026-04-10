//
// Created by hzw on 2026/3/29.
//
#ifndef MY_INFERENCE_KERNEL_KEY_UTIL_H
#define MY_INFERENCE_KERNEL_KEY_UTIL_H

#include "graph/node/op_node.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/factory.h"

#define REGISTER_KERNEL_KEY_GENERATOR(op_type,kernel_key_generator) GENERIC_REGISTER(my_inference::OpType,my_inference::KernelKeyGenerator *,op_type,kernel_key_generator)

namespace my_inference {
    KernelKey getKernelKey(const OpNode *op, DeviceType device_type,
                           isa_type isa_type);
}

#endif //MY_INFERENCE_KERNEL_KEY_UTIL_H
