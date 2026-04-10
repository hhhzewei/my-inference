//
// Created by hzw on 2026/2/17.
//
#pragma once

#include "backend/device_type.h"
#include "backend/isa_type.h"
#include "graph/node/op_node.h"

namespace my_inference {
    using KernelKey = uint64_t;

    class KernelKeyGenerator {
    public:
        virtual ~KernelKeyGenerator() = default;

        KernelKey operator()(const OpNode *op, const DeviceType device_type, const isa_type isa_type) const {
            return baseKey(device_type, isa_type, op->type(), op->dataType())
                   | reservedKey(op);
        }

    private:
        constexpr static unsigned KeyBits = 64;
        constexpr static unsigned DeviceTypeBits = 2;
        constexpr static unsigned DeviceTypeBitOffset = KeyBits - DeviceTypeBits;
        constexpr static unsigned IsaTypeBits = 8;
        constexpr static unsigned IsaTypeBitOffset = DeviceTypeBitOffset - DeviceTypeBits;
        constexpr static unsigned OpTypeBits = 10;
        constexpr static unsigned OpTypeBitOffset = IsaTypeBitOffset - OpTypeBits;
        constexpr static unsigned DataTypeBits = 5;
        constexpr static unsigned DATA_TYPE_BIT_OFFSET = DeviceTypeBitOffset - DataTypeBits;

        [[nodiscard]] virtual KernelKey reservedKey(const OpNode *op) const =0;

    protected:
        constexpr static unsigned ReservedBits = KeyBits - IsaTypeBits - DeviceTypeBits - OpTypeBits - DataTypeBits;

        constexpr static KernelKey baseKey(const DeviceType device_type, const isa_type isa_type,
                                           const OpType op_type, const DataType data_type) {
            return static_cast<KernelKey>(op_type) << OpTypeBitOffset |
                   static_cast<KernelKey>(isa_type) << IsaTypeBitOffset |
                   static_cast<KernelKey>(device_type) << DeviceTypeBitOffset |
                   static_cast<KernelKey>(data_type) << DATA_TYPE_BIT_OFFSET;
        }
    };
}
