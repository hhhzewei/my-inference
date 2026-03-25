//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
#define MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/factory.h"

namespace my_inference {
#define REGISTER_ATTR_PROPAGATOR(key,value) GENERIC_REGISTER(OpType,AttrPropagator*,key,value)

    inline void propagateAttribute(OpNode *op) {
        using AttrPropagatorFactory = GenericFactory<OpType, AttrPropagator *>;
        if (AttrPropagator *attr_propagator = AttrPropagatorFactory::instance().get(op->type())) {
            (*attr_propagator)(op);
        }
    }
}
#endif //MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
