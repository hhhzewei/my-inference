//
// Created by hzw on 2026/4/22.
//
#include "graph/node/shape_util.h"

using namespace my_inference;

std::vector<TensorDim> my_inference::shapeAlign(const std::vector<TensorDim> &shape, const int num_dim) {
    if (shape.size() >= num_dim) {
        return shape;
    }
    std::vector<TensorDim> new_shape;
    new_shape.reserve(num_dim);
    for (int i = 0; i < num_dim - shape.size(); ++i) {
        new_shape.emplace_back(1);
    }
    for (const auto &dim: shape) {
        new_shape.emplace_back(dim);
    }
    return std::move(new_shape);
}

std::vector<TensorDim> my_inference::transposeShape(const std::vector<TensorDim> &shape,
                                                    const std::vector<int64_t> &perm) {
    std::vector<TensorDim> transposed_shape;
    transposed_shape.reserve(shape.size());
    for (const auto dim_idx: perm) {
        transposed_shape.emplace_back(shape[dim_idx]);
    }
    return transposed_shape;
}
