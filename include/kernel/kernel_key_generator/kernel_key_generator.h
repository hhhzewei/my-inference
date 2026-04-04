//
// Created by hzw on 2026/2/17.
//
#pragma once

#include "graph/node/op_node.h"

namespace my_inference {
    using KernelKey = uint64_t;

    class KernelKeyGenerator {
    public:
        virtual ~KernelKeyGenerator() = default;

        KernelKey operator()(const OpNode *op) const {
            return baseKey(op->type(), op->deviceType(), op->dataType())
                   | reservedKey(op);
        }

    private:
        constexpr static unsigned KeyBits = 64;
        constexpr static unsigned OpTypeBits = 10;
        constexpr static unsigned OpTypeBitOffset = KeyBits - OpTypeBits;
        constexpr static unsigned DeviceTypeBits = 5;
        constexpr static unsigned DeviceTypeBitOffset = OpTypeBitOffset - DeviceTypeBits;
        constexpr static unsigned DataTypeBits = 5;
        constexpr static unsigned DATA_TYPE_BIT_OFFSET = DeviceTypeBitOffset - DataTypeBits;

        [[nodiscard]] virtual KernelKey reservedKey(const OpNode *op) const =0;

    protected:
        constexpr static unsigned ReservedBits = KeyBits - OpTypeBits - DeviceTypeBits - DataTypeBits;

        constexpr static KernelKey baseKey(const OpType &op_type, const DeviceType &device_type,
                                           const DataType &data_type) {
            return static_cast<KernelKey>(op_type) << OpTypeBitOffset |
                   static_cast<KernelKey>(device_type) << DeviceTypeBitOffset |
                   static_cast<KernelKey>(data_type) << DATA_TYPE_BIT_OFFSET;
        }
    };
}
