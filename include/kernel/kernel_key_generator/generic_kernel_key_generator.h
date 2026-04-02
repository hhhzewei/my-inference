//
// Created by hzw on 2026/4/1.
//

#ifndef MY_INFERENCE_GENERIC_KERNEL_KEY_GENERATOR_H
#define MY_INFERENCE_GENERIC_KERNEL_KEY_GENERATOR_H
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/singleton.h"

namespace my_inference {
    class GenericKernelKeyGenerator : public KernelKeyGenerator, public Singleton<GenericKernelKeyGenerator> {
    public:
        // 用于注册
        constexpr static KernelKey generate(const OpType &op_type, const DeviceType &device_type,
                                            const DataType &data_type) {
            return baseKey(op_type, device_type, data_type);
        }

    private:
        [[nodiscard]] KernelKey reservedKey(const OpNode *op_node) const override {
            return 0;
        }
    };
}
#endif //MY_INFERENCE_GENERIC_KERNEL_KEY_GENERATOR_H
