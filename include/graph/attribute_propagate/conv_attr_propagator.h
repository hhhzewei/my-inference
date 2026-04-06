//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
#define MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/singleton.h"

namespace my_inference {
    class ConvAttrPropagator : public AttrPropagator, public Singleton<ConvAttrPropagator> {
        DECLARE_SINGLETON(ConvAttrPropagator)

    public:
        void operator()(OpNode *) override;
    };
}

#endif //MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
