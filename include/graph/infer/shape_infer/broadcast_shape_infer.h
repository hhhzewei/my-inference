//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#define MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#include "graph/infer/shape_infer.h"

namespace my_inference {
    class BroadcastShapeInfer : public ShapeInfer {
    public:
        static BroadcastShapeInfer *instance() {
            static BroadcastShapeInfer instance_;
            return &instance_;
        }

        void operator()(OpNode *) override;

    private:
        BroadcastShapeInfer() = default;
    };
}
#endif //MY_INFERENCE_BROADCAST_SHAPE_INFER_H
