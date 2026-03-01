//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_ATTR_PROPAGATOR_H
#define MY_INFERENCE_ATTR_PROPAGATOR_H
#include "graph/node/op_node.h"

namespace my_inference {
    class AttrPropagator {
    public:
        virtual ~AttrPropagator() = default;

        virtual void operator()(OpNode *) =0;

    protected:
        template<typename T>
        static void SetDefaultAttr(OpNode *op, const AttributeKey key, const T &default_value) {
            if (!op->hasAttribute(key)) {
                op->setAttribute(key, default_value);
            }
        }
    };
}

#endif //MY_INFERENCE_ATTR_PROPAGATOR_H
