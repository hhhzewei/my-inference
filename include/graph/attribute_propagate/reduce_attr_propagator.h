//
// Created by hzw on 2026/4/6.
//

#ifndef MY_INFERENCE_REDUCE_ATTR_PROPAGATOR_H
#define MY_INFERENCE_REDUCE_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/singleton.h"

namespace my_inference {
    class ReduceAttrPropagator : public AttrPropagator, public Singleton<ReduceAttrPropagator> {
        DECLARE_SINGLETON(ReduceAttrPropagator)

    public:
        void operator()(OpNode *op) override;
    };
}
#endif //MY_INFERENCE_REDUCE_ATTR_PROPAGATOR_H
