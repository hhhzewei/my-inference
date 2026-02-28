//
// Created by hzw on 2026/2/28.
//
#pragma once

#include <vector>
#include "graph/node/tensor_dim.h"

namespace my_inference {
    std::vector<TensorDim> default_stride(const std::vector<TensorDim> &shape);

    std::vector<TensorDim> broadcast_stride(const std::vector<TensorDim> &shape,
                                            const std::vector<TensorDim> &expected_shape);
}
