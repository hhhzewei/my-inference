//
// Created by hzw on 2026/3/29.
//

#include "optimize/constant_folder/elementwise_folder.h"
#include "optimize/constant_folder/op_folder_util.h"
#include "util/math.h"

using namespace my_inference;
REGISTER_OP_FOLDER(getFolderKey(OpType::Add, DataType::Float32),
                   (&ElementwiseFolder<float, AddFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Sub, DataType::Float32),
                   (&ElementwiseFolder<float, SubFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Mul, DataType::Float32),
                   (&ElementwiseFolder<float, MulFunctor<float> >::instance()));
REGISTER_OP_FOLDER(getFolderKey(OpType::Div, DataType::Float32),
                   (&ElementwiseFolder<float, DivFunctor<float> >::instance()));
