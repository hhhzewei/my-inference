//
// Created by hzw on 2026/3/1.
//

#ifndef MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
#define MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/Singleton.h"

namespace my_inference {
    class GemmAttrPropagator : public AttrPropagator, public Singleton<GemmAttrPropagator> {
        DECLARE_SINGLETON(GemmAttrPropagator)

    public:
        void operator()(OpNode *) override;

    private:
        static constexpr int64_t DEFAULT_TRANS_A = false;
        static constexpr int64_t DEFAULT_TRANS_B = false;
    };
}
#endif //MY_INFERENCE_GEMM_ATTR_PROPAGATOR_H
