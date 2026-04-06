//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_REDUCE_SHAPE_INFER_H
#define MY_INFERENCE_REDUCE_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"
#include "util/singleton.h"

namespace my_inference {
    class ReduceShapeInfer : public ShapeInfer, public Singleton<ReduceShapeInfer> {
        DECLARE_SINGLETON(ReduceShapeInfer);

    public:
        void operator()(OpNode *) override;
    };
}
#endif //MY_INFERENCE_REDUCE_SHAPE_INFER_H
