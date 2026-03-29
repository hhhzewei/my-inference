//
// Created by hzw on 2026/3/29.
//

#include "optimize/constant_folder/element_wise_folder.h"
#include "optimize/constant_folder/op_folder_util.h"
#include "util/math.h"

using namespace my_inference;
REGISTER_OP_FOLDER(getFolderKey(OpType::Add, DataType::Float32, DeviceType::CPU),
                   (&ElementWiseFolder<float, AddFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Sub, DataType::Float32, DeviceType::CPU),
                   (&ElementWiseFolder<float, SubFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Mul, DataType::Float32, DeviceType::CPU),
                   (&ElementWiseFolder<float, MulFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Div, DataType::Float32, DeviceType::CPU),
                   (&ElementWiseFolder<float, DivFunctor<float> >::instance()));
