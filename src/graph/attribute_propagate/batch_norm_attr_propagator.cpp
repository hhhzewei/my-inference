//
// Created by hzw on 2026/4/1.
//

#include "graph/attribute_propagate/batch_norm_attr_propagator.h"
#include "graph/attribute_propagate/attr_propagate_util.h"

using namespace my_inference;
REGISTER_ATTR_PROPAGATOR(OpType::BatchNormalization, &BatchNormAttrPropagator::instance());

void BatchNormAttrPropagator::operator()(OpNode *op) {
    SetDefaultAttr<float>(op, AttributeKey::Epsilon, 0.0f);
}
