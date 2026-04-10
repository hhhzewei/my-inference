//
// Created by hzw on 2026/2/27.
//

#ifndef MY_INFERENCE_OP_FOLDER_UTIL_H
#define MY_INFERENCE_OP_FOLDER_UTIL_H
#include "backend/device_type.h"
#include "optimize/constant_folder/op_folder.h"
#include "util/factory.h"

namespace my_inference {
    using FolderKey = uint32_t;

#define REGISTER_OP_FOLDER(folder_key,op_folder) GENERIC_REGISTER(FolderKey,OpFolder *,folder_key,op_folder)

    FolderKey getFolderKey(OpType type, const DataType &data_type);

    FolderKey getFolderKey(const OpNode *op);

    bool opFold(OpNode *op);
}
#endif //MY_INFERENCE_OP_FOLDER_UTIL_H
