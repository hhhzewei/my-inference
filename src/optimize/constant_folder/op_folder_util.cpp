//
// Created by hzw on 2026/3/29.
//
#include "optimize/constant_folder/op_folder_util.h"
#include "graph/node/tensor_node.h"

my_inference::FolderKey my_inference::getFolderKey(const OpType type, const DataType &data_type,
                                                   const DeviceType &device_type) {
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

my_inference::FolderKey my_inference::getFolderKey(const OpNode *op) {
    return getFolderKey(op->type(), op->input(0)->dataType(), op->deviceType());
}

bool my_inference::opFold(OpNode *op) {
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
