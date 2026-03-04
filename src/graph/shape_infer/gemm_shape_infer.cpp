//
// Created by hzw on 2026/2/27.
//

#include "graph/shape_infer/gemm_shape_infer.h"
#include "graph/node/tensor_node.h"
#include "graph/node/attribute/attribute_key.h"

void my_inference::GemmShapeInfer::operator()(OpNode *op) {
    const bool transA = op->attribute<int64_t>(AttributeKey::TransA).value();
    const bool transB = op->attribute<int64_t>(AttributeKey::TransB).value();
    const auto &a_shape = op->input(0)->shape();
    const auto &b_shape = op->input(1)->shape();
    const TensorDim &M = transA ? a_shape[1] : a_shape[0];
    const TensorDim &N = transB ? b_shape[0] : b_shape[1];
    op->output(0)->setShape({M, N});
}
