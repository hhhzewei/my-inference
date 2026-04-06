//
// Created by hzw on 2026/4/5.
//

#include "graph/shape_infer/reduce_shape_infer.h"
#include "graph/node/tensor_node.h"
#include "graph/shape_infer/shape_infer_util.h"

using namespace my_inference;

REGISTER_SHAPE_INFER(OpType::ReduceMax,&ReduceShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::ReduceMin,&ReduceShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::ReduceMean,&ReduceShapeInfer::instance());
REGISTER_SHAPE_INFER(OpType::ReduceSum,&ReduceShapeInfer::instance());

void ReduceShapeInfer::operator()(OpNode *op) {
    const bool keep_dim = op->attribute<int64_t>(AttributeKey::KeepDims).value();
    const auto axes = op->attribute<std::vector<int64_t> >(AttributeKey::Axes).value();
    std::vector<TensorDim> expected_shape;
    expected_shape.reserve(op->input(0)->numDim());
    auto &raw_shape = op->input(0)->shape();
    for (int i = 0; i < raw_shape.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            expected_shape.emplace_back(raw_shape[i]);
        } else if (keep_dim) {
            expected_shape.emplace_back(1);
        }
    }
    op->output(0)->setShape(std::move(expected_shape));
}
