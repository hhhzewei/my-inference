//
// Created by hzw on 2026/2/27.
//

#include "graph/infer/shape_infer/gemm_shape_infer.h"

#include "graph/node/attribute_key.h"

void my_inference::GemmShapeInfer::operator()(OpNode *op) {
    const bool transA = op->attribute<int64_t>(attribute_key::TransA, 0);
    const bool transB = op->attribute<int64_t>(attribute_key::TransB, 0);
    const auto &a_shape = op->input(0)->shape();
    const auto &b_shape = op->input(1)->shape();
    const TensorDim &M = transA ? a_shape[1] : a_shape[0];
    const TensorDim &N = transB ? b_shape[0] : b_shape[1];
    op->output(0)->setShape({M, N});
}
