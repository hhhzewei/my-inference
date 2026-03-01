//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
#define MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"

namespace my_inference {
    class ConvAttrPropagator : public AttrPropagator {
    public:
        static ConvAttrPropagator *instance() {
            static ConvAttrPropagator instance_;
            return &instance_;
        }

        void operator()(OpNode *) override;

    private:
        static constexpr int64_t DEFAULT_PAD = 0;
        static constexpr int64_t DEFAULT_STRIDE = 1;
        static constexpr int64_t DEFAULT_DIALATION = 1;

        ConvAttrPropagator() = default;
    };
}

#endif //MY_INFERENCE_CONV_ATTR_PROPAGATOR_H
