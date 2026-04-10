//
// Created by hzw on 2026/4/4.
//

#include "kernel/kernel_key_generator/conv_key_generator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"

using namespace my_inference;

REGISTER_KERNEL_KEY_GENERATOR(OpType::Conv, &ConvKeyGenerator::instance());

KernelKey ConvKeyGenerator::generate(const DeviceType device_type, const IsaType isa_type, const OpType op_type,
                                          const DataType data_type, const int num_dim, const ConvType conv_type) {
    return baseKey(device_type, isa_type, op_type, data_type) | reservedKey(num_dim, conv_type);
}

KernelKey ConvKeyGenerator::reservedKey(const int num_dim, const ConvType conv_type) {
    constexpr unsigned NumDimBits = 2;
    constexpr unsigned NumDimOffset = ReservedBits - NumDimBits;
    constexpr unsigned ConvTypeBits = 2;
    constexpr unsigned ConvTypeOffset = NumDimOffset - ConvTypeBits;
    return static_cast<KernelKey>(num_dim) << NumDimOffset |
           static_cast<KernelKey>(conv_type) << ConvTypeOffset;
}

KernelKey ConvKeyGenerator::reservedKey(const OpNode *op) const {
    const int num_dim = op->input(0)->numDim() - 2;
    const int group = op->attribute<int64_t>(AttributeKey::Group).value();
    const int in_channel = op->input(0)->dim(1).value();
    ConvType conv_type;
    if (group == 1) {
        conv_type = ConvType::Standard;
    } else if (group == in_channel) {
        conv_type = ConvType::Depthwise;
    } else {
        conv_type = ConvType::Grouped;
    }
    return reservedKey(num_dim, conv_type);
}
