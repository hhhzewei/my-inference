//
// Created by hzw on 2026/2/24.
//

#ifndef MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#define MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#include "graph/infer/shape_infer.h"

namespace my_inference {
    class IdentityShapeInfer : public ShapeInfer {
    public:
        static IdentityShapeInfer *instance() {
            static IdentityShapeInfer instance_;
            return &instance_;
        }

        void operator()(OpNode *) override;

    private:
        IdentityShapeInfer() = default;
    };
}

#endif //MY_INFERENCE_IDENTITY_SHAPE_INFER_H
