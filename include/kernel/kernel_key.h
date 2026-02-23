//
// Created by hzw on 2026/2/17.
//
#pragma once

#include "graph/data_type.h"
#include "graph/op_node.h"
#include "graph/op_type.h"

namespace my_inference {
    using KernelKey = uint64_t;

    class KernelKeyGenerator {
    public:
        virtual ~KernelKeyGenerator() = default;

        static KernelKeyGenerator *instance() {
            static KernelKeyGenerator instance_;
            return &instance_;
        }

        // 用于注册
        constexpr static KernelKey generate(const OpType &op_type, const DeviceType &device_type,
                                            const DataType &data_type) {
            return baseKey(op_type, device_type, data_type);
        }

        KernelKey operator()(const OpNode &op_node) const {
            return baseKey(op_node.type(), op_node.deviceType(), op_node.dataType())
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
        constexpr static unsigned RESERVED_BITS = KEY_BITS - OP_TYPE_BITS - DEVICE_TYPE_BITS - DATA_TYPE_BITS;
        constexpr static unsigned RESERVED_BIT_OFFSET = 0;

        constexpr static KernelKey baseKey(const OpType &op_type, const DeviceType &device_type,
                                           const DataType &data_type) {
            return static_cast<KernelKey>(op_type) << OP_TYPE_BIT_OFFSET |
                   static_cast<KernelKey>(device_type) << DEVICE_TYPE_BIT_OFFSET |
                   static_cast<KernelKey>(data_type) << DATA_TYPE_BIT_OFFSET;
        }

        [[nodiscard]] virtual KernelKey reservedKey(const OpNode &op_node) const {
            return 0;
        }
    };

    inline KernelKey getKernelKey(const OpNode &op_node) {
        const static std::map<OpType, KernelKeyGenerator *> map = {
            {OpType::Add, KernelKeyGenerator::instance()},
            {OpType::Sub, KernelKeyGenerator::instance()},
            {OpType::Mul, KernelKeyGenerator::instance()},
            {OpType::Div, KernelKeyGenerator::instance()},
        };
        const auto it = map.find(op_node.type());
        if (it == map.end()) {
            std::cout << "Cant find OpKeyGenerator";
            return 0;
        }
        return (*it->second)(op_node);
    }
}
