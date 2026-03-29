//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#define MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"
#include "util/singleton.h"

namespace my_inference {
    class BroadcastShapeInfer : public ShapeInfer, public Singleton<BroadcastShapeInfer> {
        DECLARE_SINGLETON(BroadcastShapeInfer)

    public:
        void operator()(OpNode *) override;
    };
}
#endif //MY_INFERENCE_BROADCAST_SHAPE_INFER_H
