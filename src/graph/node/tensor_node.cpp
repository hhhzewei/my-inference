//
// Created by hzw on 2026/3/9.
//

#include "graph/node/tensor_node.h"
#include "graph/node/op_node.h"

bool my_inference::TensorNode::isConstant() const {
    return producer_->type() == OpType::Constant;
}
