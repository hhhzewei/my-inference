//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
#define MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
#include "kernel_key_generator.h"
#include "util/singleton.h"

namespace my_inference {
    class ElementWiseKeyGenerator : public KernelKeyGenerator, public Singleton<ElementWiseKeyGenerator> {
        DECLARE_SINGLETON(ElementWiseKeyGenerator);

    public:
        static KernelKey generate(const OpType &op_type, const DeviceType &device_type, const DataType &data_type,
                                  bool isBroadcast);

    private:
        constexpr static int IS_BROADCAST_BITS = 1;
        constexpr static int IS_BROADCAST_OFFSET = RESERVED_BITS - IS_BROADCAST_BITS;

        static KernelKey reservedKey(const bool isBroadcast);

        [[nodiscard]] KernelKey reservedKey(const OpNode *op) const override;
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_KERNEL_KEY_GENERATOR_H
