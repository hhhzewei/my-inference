//
// Created by hzw on 2026/2/24.
//

#ifndef MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#define MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"
#include "util/singleton.h"

namespace my_inference {
    class IdentityShapeInfer : public ShapeInfer, public Singleton<IdentityShapeInfer> {
        DECLARE_SINGLETON(IdentityShapeInfer)

    public:
        void operator()(OpNode *) override;
    };
}

#endif //MY_INFERENCE_IDENTITY_SHAPE_INFER_H
