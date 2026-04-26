//
// Created by hzw on 2026/4/22.
//

#ifndef MY_INFERENCE_SHAPE_UTIL_H
#define MY_INFERENCE_SHAPE_UTIL_H
#include <vector>

#include "graph/node/tensor_dim.h"
namespace my_inference {
    std::vector<TensorDim> shapeAlign(const std::vector<TensorDim> &shape,int num_dim);

    std::vector<TensorDim> transposeShape(const std::vector<TensorDim> &shape,
                                                        const std::vector<int64_t> &perm);
}

#endif //MY_INFERENCE_SHAPE_UTIL_H