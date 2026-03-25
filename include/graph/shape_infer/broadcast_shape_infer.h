//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#define MY_INFERENCE_BROADCAST_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer_util.h"

namespace my_inference {
    class BroadcastShapeInfer : public ShapeInfer, public Singleton<BroadcastShapeInfer> {
        DECLARE_SINGLETON(BroadcastShapeInfer)

    public:
        void operator()(OpNode *) override;
    };

    // two math
    REGISTER_SHAPE_INFER(OpType::Add, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Sub, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Mul, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Div, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Pow, &BroadcastShapeInfer::instance());

    // logic
    REGISTER_SHAPE_INFER(OpType::Less, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Greater, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Equal, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::NotEqual, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::And, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Or, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Xor, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Max, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Min, &BroadcastShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Where, &BroadcastShapeInfer::instance());
}
#endif //MY_INFERENCE_BROADCAST_SHAPE_INFER_H
