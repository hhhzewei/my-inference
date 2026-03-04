//
// Created by hzw on 2026/2/27.
//

#ifndef MY_INFERENCE_OP_FOLDER_UTIL_H
#define MY_INFERENCE_OP_FOLDER_UTIL_H
#include "optimize/constant_folder/element_wise_folder.h"
#include "optimize/constant_folder/op_folder.h"
#include "util/math.h"

namespace my_inference {
    using FolderKey = uint32_t;

    inline FolderKey getFolderKey(const OpType type, const DataType &data_type, const DeviceType &device_type) {
        constexpr unsigned KEY_BITS = sizeof(FolderKey) * 8;
        constexpr unsigned OP_TYPE_BITS = 10;
        constexpr unsigned OP_TYPE_BIT_OFFSET = KEY_BITS - OP_TYPE_BITS;
        constexpr unsigned DEVICE_TYPE_BITS = 5;
        constexpr unsigned DEVICE_TYPE_BIT_OFFSET = OP_TYPE_BIT_OFFSET - DEVICE_TYPE_BITS;
        constexpr unsigned DATA_TYPE_BITS = 5;
        constexpr unsigned DATA_TYPE_BIT_OFFSET = DEVICE_TYPE_BIT_OFFSET - DATA_TYPE_BITS;
        return static_cast<FolderKey>(type) << OP_TYPE_BIT_OFFSET |
               static_cast<FolderKey>(device_type) << DEVICE_TYPE_BIT_OFFSET |
               static_cast<FolderKey>(data_type) << DATA_TYPE_BIT_OFFSET;
    }

    inline FolderKey getFolderKey(const OpNode *op) {
        return getFolderKey(op->type(), op->input(0)->dataType(), op->deviceType());
    }

    inline bool opFold(OpNode *op) {
        static std::map<FolderKey, OpFolder *> map = {
            {
                getFolderKey(OpType::Add, DataType::Float32, DeviceType::CPU),
                ElementWiseFolder<float, AddFunctor<float> >::instance(),
            },
            {
                getFolderKey(OpType::Sub, DataType::Float32, DeviceType::CPU),
                ElementWiseFolder<float, SubFunctor<float> >::instance(),
            },
            {
                getFolderKey(OpType::Mul, DataType::Float32, DeviceType::CPU),
                ElementWiseFolder<float, MulFunctor<float> >::instance(),
            },
            {
                getFolderKey(OpType::Div, DataType::Float32, DeviceType::CPU),
                ElementWiseFolder<float, DivFunctor<float> >::instance(),
            }
        };
        const FolderKey key = getFolderKey(op);
        const auto it = map.find(key);
        if (it == map.end()) {
            std::cout << "Cant find folder" << std::endl;
            return false;
        }
        (*it->second)(op);
        return true;
    }
}
#endif //MY_INFERENCE_OP_FOLDER_UTIL_H
