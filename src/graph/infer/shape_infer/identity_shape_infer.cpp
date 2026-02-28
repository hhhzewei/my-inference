//
// Created by hzw on 2026/2/24.
//
#include "graph/infer/shape_infer/identity_shape_infer.h"

#include "graph/infer/shape_infer/stride.h"

using namespace my_inference;

void IdentityShapeInfer::operator()(OpNode *op) {
    const std::vector<TensorDim> &expected_shape = op->input(0)->shape();
    for (TensorNode *output: op->outputs()) {
        output->setShape(expected_shape);
    }
    const std::vector<TensorDim> stride = default_stride(expected_shape);
    op->setInputsStrides(std::vector(op->numInput(), stride));
    op->setOutputsStrides(std::vector(op->numOutput(), stride));
}
