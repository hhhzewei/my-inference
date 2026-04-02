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

        KernelKey operator()(const OpNode *op_node) const {
            return baseKey(op_node->type(), op_node->deviceType(), op_node->dataType())
                   | reservedKey(op_node);
        }

    private:
        constexpr static unsigned KEY_BITS = 64;
        constexpr static unsigned OP_TYPE_BITS = 10;
        constexpr static unsigned OP_TYPE_BIT_OFFSET = KEY_BITS - OP_TYPE_BITS;
        constexpr static unsigned DEVICE_TYPE_BITS = 5;
        constexpr static unsigned DEVICE_TYPE_BIT_OFFSET = OP_TYPE_BIT_OFFSET - DEVICE_TYPE_BITS;
        constexpr static unsigned DATA_TYPE_BITS = 5;
        constexpr static unsigned DATA_TYPE_BIT_OFFSET = DEVICE_TYPE_BIT_OFFSET - DATA_TYPE_BITS;

        [[nodiscard]] virtual KernelKey reservedKey(const OpNode *op_node) const =0;

    protected:
        constexpr static unsigned RESERVED_BITS = KEY_BITS - OP_TYPE_BITS - DEVICE_TYPE_BITS - DATA_TYPE_BITS;

        constexpr static KernelKey baseKey(const OpType &op_type, const DeviceType &device_type,
                                           const DataType &data_type) {
            return static_cast<KernelKey>(op_type) << OP_TYPE_BIT_OFFSET |
                   static_cast<KernelKey>(device_type) << DEVICE_TYPE_BIT_OFFSET |
                   static_cast<KernelKey>(data_type) << DATA_TYPE_BIT_OFFSET;
        }
    };
}
