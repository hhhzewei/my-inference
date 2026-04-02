//
// Created by hzw on 2026/4/1.
//

#ifndef MY_INFERENCE_BATCH_NORM_ATTR_PROPAGATOR_H
#define MY_INFERENCE_BATCH_NORM_ATTR_PROPAGATOR_H
#include "graph/attribute_propagate/attr_propagator.h"
#include "util/singleton.h"


class BatchNormAttrPropagator : public my_inference::AttrPropagator,
                                public my_inference::Singleton<BatchNormAttrPropagator> {
public:
    void operator()(my_inference::OpNode *) override;

private:
    constexpr static float DEFAULT_EPSILON = 0.0f;
};


#endif //MY_INFERENCE_BATCH_NORM_ATTR_PROPAGATOR_H
