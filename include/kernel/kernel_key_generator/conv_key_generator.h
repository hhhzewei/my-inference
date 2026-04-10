//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CONV_KEY_GENERATOR_H
#define MY_INFERENCE_CONV_KEY_GENERATOR_H
#include "graph/node/tensor_node.h"
#include "kernel/kernel_util.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/singleton.h"

namespace my_inference {
    class ConvKeyGenerator : public KernelKeyGenerator, public Singleton<ConvKeyGenerator> {
        DECLARE_SINGLETON(ConvKeyGenerator);

    public:
        static KernelKey generate(DeviceType device_type, IsaType isa_type, OpType op_type, DataType data_type,
                                  int num_dim, ConvType conv_type);

    private:
        static KernelKey reservedKey(int num_dim, ConvType conv_type);

        [[nodiscard]] KernelKey reservedKey(const OpNode *op) const override;
    };
}
#endif //MY_INFERENCE_CONV_KEY_GENERATOR_H
