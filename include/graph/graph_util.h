//
// Created by hzw on 2026/3/4.
//

#ifndef MY_INFERENCE_GRAPH_UTIL_H
#define MY_INFERENCE_GRAPH_UTIL_H
#include <memory>

#include "graph/node/tensor_node.h"
#include "graph/node/op_node.h"

namespace my_inference {
    inline auto EmptyTensor = std::make_unique<TensorNode>(
        "__EMPTY_TENSOR__", 0, std::vector<TensorDim>{}, DataType::Unknown, true);
}
#endif //MY_INFERENCE_GRAPH_UTIL_H
