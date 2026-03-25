//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
#define MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
#include "graph/node/tensor_node.h"
#include "kernel_key_generator.h"

namespace my_inference {
    class ElementWiseKeyGenerator : public KernelKeyGenerator, public Singleton<ElementWiseKeyGenerator> {
        DECLARE_SINGLETON(ElementWiseKeyGenerator);

    public:
        static KernelKey generate(const OpType &op_type, const DeviceType &device_type, const DataType &data_type,
                                  const bool isBroadcast) {
            return baseKey(op_type, device_type, data_type) || reservedKey(isBroadcast);
        }

    private:
        constexpr static int IS_BROADCAST_BITS = 1;
        constexpr static int IS_BROADCAST_OFFSET = RESERVED_BITS - IS_BROADCAST_BITS;

        static KernelKey reservedKey(const bool isBroadcast) {
            return static_cast<uint64_t>(isBroadcast) << IS_BROADCAST_OFFSET;
        }

        [[nodiscard]] KernelKey reservedKey(const OpNode *op) const override {
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
    };

    REGISTER_KERNEL_KEY_GENERATOR(OpType::Add, &ElementWiseKeyGenerator::instance());
    REGISTER_KERNEL_KEY_GENERATOR(OpType::Sub, &ElementWiseKeyGenerator::instance());
    REGISTER_KERNEL_KEY_GENERATOR(OpType::Mul, &ElementWiseKeyGenerator::instance());
    REGISTER_KERNEL_KEY_GENERATOR(OpType::Div, &ElementWiseKeyGenerator::instance());
}
#endif //MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
