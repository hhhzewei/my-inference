//
// Created by hzw on 2026/3/4.
//

#ifndef MY_INFERENCE_GRAPH_UTIL_H
#define MY_INFERENCE_GRAPH_UTIL_H

#include "graph/node/tensor_node.h"
#include "graph/node/op_node.h"

namespace my_inference {
    inline void unlinkInputOfOp(const OpNode *op) {
        for (TensorNode *input: op->inputs()) {
            input->removeConsumer(op);
        }
    }
}
#endif //MY_INFERENCE_GRAPH_UTIL_H
