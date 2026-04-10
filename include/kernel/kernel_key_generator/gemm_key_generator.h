//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_GEMM_KEY_GENERATOR_H
#define MY_INFERENCE_GEMM_KEY_GENERATOR_H
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/singleton.h"

namespace my_inference {
    class GemmKeyGenerator : public KernelKeyGenerator, public Singleton<GemmKeyGenerator> {
        DECLARE_SINGLETON(GemmKeyGenerator);

    public:
        static KernelKey generate(DeviceType device_type, IsaType isa_type, OpType op_type,
                                  DataType data_type, bool transA, bool transB);

    private:
        static KernelKey reservedKey(bool transA, bool transB);

        [[nodiscard]] KernelKey reservedKey(const OpNode *op) const override;
    };
}
#endif //MY_INFERENCE_GEMM_KEY_GENERATOR_H
