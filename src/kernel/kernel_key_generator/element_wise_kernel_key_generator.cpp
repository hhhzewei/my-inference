//
// Created by hzw on 2026/3/29.
//
#include "kernel/kernel_key_generator/kernel_key_util.h"
#include "kernel/kernel_key_generator/element_wise_kernel_key_generator.h"

using namespace my_inference;

REGISTER_KERNEL_KEY_GENERATOR(OpType::Add, &ElementWiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Sub, &ElementWiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Mul, &ElementWiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Div, &ElementWiseKeyGenerator::instance());

KernelKey ElementWiseKeyGenerator::generate(const OpType &op_type, const DeviceType &device_type,
    const DataType &data_type, const bool isBroadcast) {
    return baseKey(op_type, device_type, data_type) || reservedKey(isBroadcast);
}

KernelKey ElementWiseKeyGenerator::reservedKey(const bool isBroadcast) {
    return static_cast<uint64_t>(isBroadcast) << IS_BROADCAST_OFFSET;
}

KernelKey ElementWiseKeyGenerator::reservedKey(const OpNode *op) const {
    bool isBroadcast = false;
    for (int i = 0; i < op->numInput(); ++i) {
        for (auto &stride: op->inputStrides(i)) {
            if (!stride.isValue() || stride.value() != 0) continue;
            isBroadcast = true;
            break;
        }
    }
    return reservedKey(isBroadcast);
}
