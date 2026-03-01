#include "graph/attribute_propagate/gemm_attr_propagator.h"

#include "graph/shape_infer/conv_shape_infer.h"
//
// Created by hzw on 2026/3/1.
//
void my_inference::GemmAttrPropagator::operator()(OpNode *op) {
    SetDefaultAttr(op, AttributeKey::TransA, DEFAULT_TRANS_A);
    SetDefaultAttr(op, AttributeKey::TransB, DEFAULT_TRANS_B);
}
