//
// Created by hzw on 2026/2/25.
//
#include "graph/infer/shape_infer/broadcast_shape_infer.h"

using namespace my_inference;

void BroadcastShapeInfer::operator()(OpNode *op) {
    size_t numDim = 0;
    // 统计标准维数
    for (const TensorNode *input: op->inputs()) {
        numDim = std::max(numDim, input->numDim());
    }
    // 生成最终结果
    std::vector expected_shape(numDim, TensorDim(1));
    for (int i = 0; i < numDim; ++i) {
        TensorDim expected_dim(1);
        for (TensorNode *input: op->inputs()) {
            TensorDim dim(1);
            if (input->numDim() >= i + 1) {
                dim = input->dim(input->numDim() - 1 - i);
            }
            // 校验
            if (expected_dim.isClear()) {
                if (dim.isClear() && expected_dim != dim) {
                    std::cout << "Dim error";
                    return;
                }
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
}
