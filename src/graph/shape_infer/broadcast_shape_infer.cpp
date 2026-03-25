//
// Created by hzw on 2026/2/25.
//
#include "graph/shape_infer/broadcast_shape_infer.h"
#include "graph/node/tensor_node.h"
#include "graph/shape_infer/stride.h"

using namespace my_inference;

void BroadcastShapeInfer::operator()(OpNode *op) {
    int numDim = 0;
    // 统计标准维数
    for (const TensorNode *input: op->inputs()) {
        numDim = std::max(numDim, input->numDim());
    }
    // 生成最终结果
    std::vector expected_shape(numDim, TensorDim(1));
    for (int i = 0; i < numDim; ++i) {
        TensorDim expected_dim(1);
        for (const TensorNode *input: op->inputs()) {
            TensorDim dim(1);
            if (const int idx = input->numDim() - 1 - i; idx >= 0) {
                dim = input->dim(idx);
            }
            // 校验
            if (expected_dim.isClear() && dim.isClear() && expected_dim != dim) {
                std::cout << "Dim error";
                return;
            }
            if (dim.isClear()) {
                expected_dim = dim;
            }
        }
        //reverse iterate so cant push_back/emplace_back
        expected_shape[numDim - 1 - i] = expected_dim;
    }
    for (TensorNode *output: op->outputs()) {
        output->setShape(expected_shape);
    }
    // 生成strides
    std::vector<std::vector<TensorDim> > inputs_strides;
    inputs_strides.reserve(op->numInput());
    for (const TensorNode *input: op->inputs()) {
        inputs_strides.emplace_back(broadcast_stride(input->shape(), expected_shape));
    }
    op->setInputsStrides(std::move(inputs_strides));
    const std::vector<TensorDim> output_strides = default_stride(expected_shape);
    op->setOutputsStrides(std::vector(op->numOutput(), output_strides));
}
