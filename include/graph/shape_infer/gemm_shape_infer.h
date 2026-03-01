//
// Created by hzw on 2026/2/27.
//

#ifndef MY_INFERENCE_GEMM_SHAPE_INFER_H
#define MY_INFERENCE_GEMM_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"

namespace my_inference {
    class GemmShapeInfer : public ShapeInfer {
    public:
        static GemmShapeInfer *instance() {
            static GemmShapeInfer instance_;
            return &instance_;
        }

        void operator()(OpNode *) override;

    private:
        GemmShapeInfer() = default;
    };
}

#endif //MY_INFERENCE_GEMM_SHAPE_INFER_H
