//
// Created by hzw on 2026/2/17.
//

#ifndef MY_INFERENCE_OP_KEY_H
#define MY_INFERENCE_OP_KEY_H

#include "graph/data_type.h"
#include "graph/op_node.h"
#include "graph/op_type.h"

using OpKey = uint64_t;

class OpKeyGenerator {
    constexpr static unsigned KEY_BITS = 64;
    constexpr static unsigned OP_TYPE_BITS = 10;
    constexpr static unsigned OP_TYPE_BIT_OFFSET = KEY_BITS - OP_TYPE_BITS;
    constexpr static unsigned DEVICE_TYPE_BITS = 5;
    constexpr static unsigned DEVICE_TYPE_BIT_OFFSET = OP_TYPE_BIT_OFFSET - DEVICE_TYPE_BITS;
    constexpr static unsigned DATA_TYPE_BITS = 5;
    constexpr static unsigned DATA_TYPE_BIT_OFFSET = DEVICE_TYPE_BIT_OFFSET - DATA_TYPE_BITS;
    constexpr static unsigned RESERVED_BITS = KEY_BITS - OP_TYPE_BITS - DEVICE_TYPE_BITS - DATA_TYPE_BITS;
    constexpr static unsigned RESERVED_BIT_OFFSET = 0;

public:
    virtual ~OpKeyGenerator() = default;

    // 用于注册
    constexpr static OpKey GetOpKey(const OpType &op_type, const DeviceType &device_type, const DataType &data_type) {
        return GetOpKeyBase(op_type, device_type, data_type);
    }

    OpKey operator()(const OpNode &op_node) const {
        return GetOpKeyBase(op_node.getType(), op_node.getDeviceType(), op_node.getDataType())
               | reservedKey(op_node);
    }

private:
    constexpr static OpKey GetOpKeyBase(const OpType &op_type, const DeviceType &device_type, const DataType &data_type) {
        return static_cast<OpKey>(op_type) << OP_TYPE_BIT_OFFSET |
               static_cast<OpKey>(device_type) << DEVICE_TYPE_BIT_OFFSET |
               static_cast<OpKey>(data_type) << DATA_TYPE_BIT_OFFSET;
    }

    virtual OpKey reservedKey(const OpNode &op_node) const {
        return 0;
    }
};

inline OpKey getOpKey(const OpNode &op_node) {
    const static std::map<OpType, OpKeyGenerator> map = {};
    const auto it = map.find(op_node.getType());
    if (it == map.end()) {
        std::cout << "Cant find OpKeyGenerator";
    }
    return it->second(op_node);
}

#endif //MY_INFERENCE_OP_KEY_H
