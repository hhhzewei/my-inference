//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
#define MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/singleton.h"

namespace my_inference {
    class GemmAttrPropagator : public AttrPropagator, public Singleton<GemmAttrPropagator> {
        DECLARE_SINGLETON(GemmAttrPropagator)

    public:
        void operator()(OpNode *) override;
    };
}
#endif //MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
