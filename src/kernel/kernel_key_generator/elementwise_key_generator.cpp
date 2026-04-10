//
// Created by hzw on 2026/3/29.
//
#include "kernel/kernel_key_generator/kernel_key_util.h"
#include "kernel/kernel_key_generator/elementwise_key_generator.h"

using namespace my_inference;

REGISTER_KERNEL_KEY_GENERATOR(OpType::Add, &ElementwiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Sub, &ElementwiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Mul, &ElementwiseKeyGenerator::instance());
REGISTER_KERNEL_KEY_GENERATOR(OpType::Div, &ElementwiseKeyGenerator::instance());

KernelKey ElementwiseKeyGenerator::generate(const DeviceType device_type, const IsaType isa_type,
                                            const OpType op_type, const DataType data_type, const bool is_broadcast) {
    return baseKey(device_type, isa_type, op_type, data_type) | reservedKey(is_broadcast);
}

KernelKey ElementwiseKeyGenerator::reservedKey(const bool isBroadcast) {
    return static_cast<uint64_t>(isBroadcast) << IsBroadcastOffset;
}

KernelKey ElementwiseKeyGenerator::reservedKey(const OpNode *op) const {
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
