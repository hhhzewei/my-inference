//
// Created by hzw on 2026/2/24.
//

#ifndef MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#define MY_INFERENCE_IDENTITY_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer_util.h"

namespace my_inference {
    class IdentityShapeInfer : public ShapeInfer, public Singleton<IdentityShapeInfer> {
        DECLARE_SINGLETON(IdentityShapeInfer)

    public:
        void operator()(OpNode *) override;
    };

    // activation
    REGISTER_SHAPE_INFER(OpType::Relu, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Sigmoid, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Tanh, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Gelu, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Softmax, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::LeakyRelu, &IdentityShapeInfer::instance());

    // one math
    REGISTER_SHAPE_INFER(OpType::Exp, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Log, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Sqrt, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Abs, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Neg, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Erf, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Not, &IdentityShapeInfer::instance());

    // convert
    REGISTER_SHAPE_INFER(OpType::Identity, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Cast, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::Clip, &IdentityShapeInfer::instance());

    // normalize
    REGISTER_SHAPE_INFER(OpType::LayerNormalization, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::BatchNormalization, &IdentityShapeInfer::instance());
    REGISTER_SHAPE_INFER(OpType::InstanceNormalization, &IdentityShapeInfer::instance());
}

#endif //MY_INFERENCE_IDENTITY_SHAPE_INFER_H
