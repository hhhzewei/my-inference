//
// Created by hzw on 2026/4/4.
//

#include "kernel/kernel_key_generator/clip_key_generator.h"

#include "kernel/kernel_key_generator/kernel_key_util.h"

using namespace my_inference;

REGISTER_KERNEL_KEY_GENERATOR(OpType::Clip, &ClipKeyGenerator::instance());

KernelKey ClipKeyGenerator::generate(const DeviceType device_type, const isa_type isa_type, const OpType op_type,
                                     const DataType data_type, const ClipType clip_type) {
    return baseKey(device_type, isa_type, op_type, data_type) | reservedKey(clip_type);
}

KernelKey ClipKeyGenerator::reservedKey(const ClipType clip_type) {
    constexpr unsigned ClipTypeBits = 1;
    constexpr unsigned ClipTypeOffset = ReservedBits - ClipTypeBits;
    return static_cast<KernelKey>(clip_type) << ClipTypeOffset;
}

KernelKey ClipKeyGenerator::reservedKey(const OpNode *op) const {
    auto clip_type = ClipType::Standard;
    if (op->input(1)->isConstant() && op->input(2)->isConstant()) {
        const DataType data_type = op->input(0)->dataType();
        void *min_ptr = op->input(1)->data(), *max_ptr = op->input(2)->data();
        switch (data_type) {
            case DataType::Float32: {
                if (*static_cast<float *>(min_ptr) == 0 && *static_cast<float *>(max_ptr) == 6) {
                    clip_type = ClipType::Relu6;
                }
                break;
            }
            default: ;
        }
    }
    return reservedKey(clip_type);
}
