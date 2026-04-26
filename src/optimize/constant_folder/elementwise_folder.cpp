//
// Created by hzw on 2026/3/29.
//

#include "optimize/constant_folder/elementwise_folder.h"
#include "optimize/constant_folder/op_folder_util.h"
#include "util/math.h"

using namespace my_inference;

REGISTER_OP_FOLDER(getFolderKey(OpType::Add, DataType::Float32),
                   (&ElementwiseFolder<float, BinaryOpType::Add >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Sub, DataType::Float32),
                   (&ElementwiseFolder<float, BinaryOpType::Sub >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Mul, DataType::Float32),
                   (&ElementwiseFolder<float, BinaryOpType::Mul >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Div, DataType::Float32),
                   (&ElementwiseFolder<float, BinaryOpType::Div >::instance()));
