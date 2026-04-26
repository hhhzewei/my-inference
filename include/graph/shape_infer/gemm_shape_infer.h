//
// Created by hzw on 2026/2/27.
//

#ifndef MY_INFERENCE_GEMM_SHAPE_INFER_H
#define MY_INFERENCE_GEMM_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"
#include "util/singleton.h"

namespace my_inference {
    class GemmShapeInfer : public ShapeInfer, public Singleton<GemmShapeInfer> {
        DECLARE_SINGLETON(GemmShapeInfer);

    public:
        void operator()(OpNode *) override;
    };
}

#endif //MY_INFERENCE_GEMM_SHAPE_INFER_H
