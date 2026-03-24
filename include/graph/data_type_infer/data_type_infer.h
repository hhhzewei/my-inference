//
// Created by hzw on 2026/2/24.
//
#pragma once
#include "graph/node/data_type.h"
#include "graph/node/op_node.h"

namespace my_inference {
    void inferDataType(const OpNode *op);

    void inferDataType(const OpNode *op, DataType target);
}
