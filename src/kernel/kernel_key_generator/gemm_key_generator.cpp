//
// Created by hzw on 2026/4/5.
//

#include "kernel/kernel_key_generator/gemm_key_generator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"

using namespace my_inference;
REGISTER_KERNEL_KEY_GENERATOR(OpType::Gemm, &GemmKeyGenerator::instance());

KernelKey GemmKeyGenerator::generate(const DeviceType device_type, const IsaType isa_type,
                                     const OpType op_type, const DataType data_type, const bool transA,
                                     const bool transB) {
    return baseKey(device_type, isa_type, op_type, data_type) |
           reservedKey(transA, transB);
}

KernelKey GemmKeyGenerator::reservedKey(const bool transA, const bool transB) {
    static constexpr unsigned TransABits = 1;
    static constexpr unsigned TransAOffset = ReservedBits - TransABits;
    static constexpr unsigned TransBBits = 1;
    static constexpr unsigned TransBOffset = TransAOffset - TransBBits;
    return static_cast<KernelKey>(transA) << TransAOffset |
           static_cast<KernelKey>(transB) << TransBOffset;
}

KernelKey GemmKeyGenerator::reservedKey(const OpNode *op) const {
    return reservedKey(op->attribute<int64_t>(AttributeKey::TransA).value(),
                       op->attribute<int64_t>(AttributeKey::TransB).value());
}
