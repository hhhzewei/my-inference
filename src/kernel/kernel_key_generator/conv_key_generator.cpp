//
// Created by hzw on 2026/4/4.
//

#include "kernel/kernel_key_generator/conv_key_generator.h"

#include "graph/node/attribute/conv_layout.h"
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
    auto input = op->input(0);
    const int num_dim = input->numDim() - 2;
    const auto layout = op->attribute<int64_t>(AttributeKey::Layout).value();
    const int in_channel = layout == ConvLayout::NHWC ? input->dim(3).value() : input->dim(1).value();
    const auto output = op->output(0);
    const int out_channel = layout == ConvLayout::NHWC ? output->dim(3).value() : output->dim(1).value();
    ConvType conv_type;
    const int group = op->attribute<int64_t>(AttributeKey::Group).value();
    if (group == 1) {
        conv_type = ConvType::Standard;
    } else if (group == in_channel && in_channel == out_channel) {
        conv_type = ConvType::Depthwise;
    } else {
        conv_type = ConvType::Grouped;
    }
    return reservedKey(num_dim, conv_type);
}
