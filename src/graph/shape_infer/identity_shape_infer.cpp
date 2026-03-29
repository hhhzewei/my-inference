//
// Created by hzw on 2026/2/24.
//
#include "graph/shape_infer/identity_shape_infer.h"
#include "graph/shape_infer/stride.h"
#include "graph/node/tensor_node.h"
#include "graph/shape_infer/shape_infer_util.h"

using namespace my_inference;


// activation
REGISTER_SHAPE_INFER(OpType::Relu, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Sigmoid, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Tanh, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Gelu, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Softmax, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::LeakyRelu, &IdentityShapeInfer::instance());

// one math
REGISTER_SHAPE_INFER(OpType::Exp, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Log, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Sqrt, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Abs, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Neg, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Erf, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Not, &IdentityShapeInfer::instance());

// convert
REGISTER_SHAPE_INFER(OpType::Identity, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Cast, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::Clip, &IdentityShapeInfer::instance());

// normalize
REGISTER_SHAPE_INFER(OpType::LayerNormalization, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::BatchNormalization, &IdentityShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::InstanceNormalization, &IdentityShapeInfer::instance());

void IdentityShapeInfer::operator()(OpNode *op) {
    const std::vector<TensorDim> &expected_shape = op->input(0)->shape();
    for (TensorNode *output: op->outputs()) {
        output->setShape(expected_shape);
    }
    const std::vector<TensorDim> stride = default_stride(expected_shape);
    op->setInputsStrides(std::vector(op->numInput(), stride));
    op->setOutputsStrides(std::vector(op->numOutput(), stride));
}
