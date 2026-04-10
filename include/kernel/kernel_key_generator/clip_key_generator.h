//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CLIP_KEY_GENERATOR_H
#define MY_INFERENCE_CLIP_KEY_GENERATOR_H
#include "graph/node/tensor_node.h"
#include "kernel/kernel_util.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/singleton.h"

namespace my_inference {
    class ClipKeyGenerator : public KernelKeyGenerator, public Singleton<ClipKeyGenerator> {
        DECLARE_SINGLETON(ClipKeyGenerator)

    public:
        static KernelKey generate(DeviceType device_type, isa_type isa_type, OpType op_type, DataType data_type, ClipType clip_type);

    private:
        static KernelKey reservedKey(ClipType clip_type);

        [[nodiscard]] KernelKey reservedKey(const OpNode *op) const override;
    };
}
#endif //MY_INFERENCE_CLIP_KEY_GENERATOR_H
