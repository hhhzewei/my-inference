//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
#define MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "graph/attribute_propagate/conv_attr_propagator.h"
#include "graph/attribute_propagate/gemm_attr_propagator.h"
#include "graph/node/op_node.h"

namespace my_inference {
    inline void propagateAttribute(OpNode *op) {
        static std::map<OpType, AttrPropagator *> map = {
            {OpType::Conv, ConvAttrPropagator::instance()},
            {OpType::Gemm, GemmAttrPropagator::instance()},
        };
        const auto it = map.find(op->type());
        if (it == map.end()) {
            return;
        }
        (*it->second)(op);
    }
}
#endif //MY_INFERENCE_ATTR_PROPAGATE_UTIL_H
