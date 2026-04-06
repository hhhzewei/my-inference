//
// Created by hzw on 2026/3/1.
//

#include "graph/attribute_propagate/gemm_attr_propagator.h"
#include "graph/attribute_propagate/attr_propagate_util.h"

REGISTER_ATTR_PROPAGATOR(my_inference::OpType::Gemm, &my_inference::GemmAttrPropagator::instance());

void my_inference::GemmAttrPropagator::operator()(OpNode *op) {
    SetDefaultAttr<int64_t>(op, AttributeKey::TransA, false);
    SetDefaultAttr<int64_t>(op, AttributeKey::TransB, false);
    SetDefaultAttr<float>(op, AttributeKey::Alpha, 1.0f);
    SetDefaultAttr<float>(op, AttributeKey::Beta, 1.0f);
}
