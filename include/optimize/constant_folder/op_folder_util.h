//
// Created by hzw on 2026/2/27.
//

#ifndef MY_INFERENCE_OP_FOLDER_UTIL_H
#define MY_INFERENCE_OP_FOLDER_UTIL_H
#include "optimize/constant_folder/op_folder.h"
#include "util/factory.h"

namespace my_inference {
    using FolderKey = uint32_t;

#define REGISTER_OP_FOLDER(folder_key,op_folder) GENERIC_REGISTER(FolderKey,OpFolder *,folder_key,op_folder)

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
        using OpFolderFactory = GenericFactory<FolderKey, OpFolder *>;
        const FolderKey key = getFolderKey(op);
        OpFolder *op_folder = OpFolderFactory::instance().get(key);
        if (!op_folder) {
            std::cout << "Cant find folder" << std::endl;
            return false;
        }
        (*op_folder)(op);
        return true;
    }
}
#endif //MY_INFERENCE_OP_FOLDER_UTIL_H
