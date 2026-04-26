//
// Created by hzw on 2026/2/28.
//
#include "graph/shape_infer/stride.h"

#include <iostream>

#include "graph/node/tensor_node.h"

using namespace my_inference;

void my_inference::initStrides(OpNode *op) {
    if (op->isConstant()) {
        return;
    }
    // input strides
    std::vector<std::vector<TensorDim> > inputs_strides;
    inputs_strides.reserve(op->numInput());
    if (isElementWise(op->type())) {
        for (const auto input: op->inputs()) {
            inputs_strides.emplace_back(broadcastStride(input->shape(), op->output(0)->shape()));
        }
    } else if (op->type() == OpType::Gemm) {
        inputs_strides.emplace_back(defaultStride(op->input(0)->shape()));
        inputs_strides.emplace_back(defaultStride(op->input(1)->shape()));
        inputs_strides.emplace_back(broadcastStride(op->input(2)->shape(), op->output(0)->shape()));
    } else {
        for (const auto input: op->inputs()) {
            inputs_strides.emplace_back(defaultStride(input->shape()));
        }
    }
    op->setInputsStrides(std::move(inputs_strides));
    // output strides
    std::vector<std::vector<TensorDim> > outputs_strides;
    outputs_strides.reserve(op->numOutput());
    for (const auto output: op->outputs()) {
        outputs_strides.emplace_back(defaultStride(output->shape()));
    }
    op->setOutputsStrides(std::move(outputs_strides));
}

std::vector<TensorDim> my_inference::defaultStride(const std::vector<TensorDim> &shape) {
    const int numDim = static_cast<int>(shape.size());
    std::vector strides(shape.size(), TensorDim(1));
    for (int i = numDim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<TensorDim> my_inference::broadcastStride(const std::vector<TensorDim> &shape,
                                                     const std::vector<TensorDim> &expected_shape) {
    const int numDim = static_cast<int>(expected_shape.size());
    std::vector<std::vector<TensorDim> > inputs_strides;
    std::vector strides(numDim, TensorDim(0));
    TensorDim stride(1);
    for (int i = 0; i < numDim; ++i) {
        const int idx = static_cast<int>(shape.size()) - 1 - i;
        TensorDim dim = idx >= 0 ? shape[idx] : TensorDim(1); // 不存在默认为1
        const int out_idx = numDim - 1 - i;
        if (expected_shape[out_idx] == dim) {
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
