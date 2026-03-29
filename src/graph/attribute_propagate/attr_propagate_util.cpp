//
// Created by hzw on 2026/3/29.
//

#include "graph/attribute_propagate/attr_propagate_util.h"

void my_inference::propagateAttribute(OpNode *op) {
    using AttrPropagatorFactory = GenericFactory<OpType, AttrPropagator *>;
    if (AttrPropagator *attr_propagator = AttrPropagatorFactory::instance().get(op->type())) {
        (*attr_propagator)(op);
    }
}
