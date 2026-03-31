//
// Created by hzw on 2026/2/28.
//
#pragma once

#include <vector>

#include "graph/node/op_node.h"
#include "graph/node/tensor_dim.h"

namespace my_inference {
    void initStrides(OpNode *op);

    std::vector<TensorDim> defaultStride(const std::vector<TensorDim> &shape);

    std::vector<TensorDim> broadcastStride(const std::vector<TensorDim> &shape,
                                            const std::vector<TensorDim> &expected_shape);
}
