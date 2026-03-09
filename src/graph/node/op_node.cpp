//
// Created by hzw on 2026/3/9.
//
#include "graph/node/tensor_node.h"
#include "graph/node/op_node.h"

int my_inference::OpNode::numConsumer() const {
    int result = 0;
    for (const auto output: outputs_) {
        result += output->numConsumer();
    }
    return result;
}
