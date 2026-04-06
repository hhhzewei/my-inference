//
// Created by hzw on 2026/4/6.
//

#include "graph/attribute_propagate/reduce_attr_propagator.h"
#include "graph/attribute_propagate/attr_propagate_util.h"
#include "graph/node/tensor_node.h"

using namespace my_inference;

REGISTER_ATTR_PROPAGATOR(OpType::ReduceMax, &ReduceAttrPropagator::instance());
REGISTER_ATTR_PROPAGATOR(OpType::ReduceMin, &ReduceAttrPropagator::instance());
REGISTER_ATTR_PROPAGATOR(OpType::ReduceMean, &ReduceAttrPropagator::instance());
REGISTER_ATTR_PROPAGATOR(OpType::ReduceSum, &ReduceAttrPropagator::instance());

void ReduceAttrPropagator::operator()(OpNode *op) {
    std::vector<int64_t> default_axes;
    default_axes.reserve(op->input(0)->numDim());
    for (int64_t i = 0; i < op->input(0)->numDim(); ++i) {
        default_axes.emplace_back(i);
    }
    SetDefaultAttr<std::vector<int64_t> >(op, AttributeKey::Axes, default_axes);
    SetDefaultAttr<int64_t>(op, AttributeKey::KeepDims, true);
}
