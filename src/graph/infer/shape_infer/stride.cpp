//
// Created by hzw on 2026/2/28.
//
#include "graph/infer/shape_infer/stride.h"

#include <iostream>

using namespace my_inference;

std::vector<TensorDim> my_inference::default_stride(const std::vector<TensorDim> &shape) {
    const int numDim = static_cast<int>(shape.size());
    std::vector strides(shape.size(), TensorDim(1));
    for (int i = numDim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<TensorDim> my_inference::broadcast_stride(const std::vector<TensorDim> &shape,
                                                      const std::vector<TensorDim> &expected_shape) {
    const int numDim = static_cast<int>(expected_shape.size());
    std::vector<std::vector<TensorDim> > inputs_strides;
    std::vector strides(numDim, TensorDim(0));
    TensorDim stride(1);
    for (int i = 0; i < numDim; ++i) {
        TensorDim dim(1); // 不存在默认为1
        if (const int idx = static_cast<int>(shape.size()) - 1 - i; idx >= 0) {
            dim = shape[idx];
        }
        const int out_idx = numDim - 1 - i;
        if (const TensorDim &out_dim = expected_shape[out_idx]; dim == out_dim) {
            strides[out_idx] = stride; //正常维度
            stride = stride * dim;
        } else if (dim.isValue() && dim.value() == 1) {
            strides[out_idx] = TensorDim(0); //广播维度
        } else {
            std::cout << "Input dim error" << std::endl;
        }
    }
    return strides;
}
