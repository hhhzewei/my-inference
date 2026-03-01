//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_SHAPE_INFER_UTIL_H
#define MY_INFERENCE_SHAPE_INFER_UTIL_H
#include "graph/infer/shape_infer.h"
#include "graph/infer/shape_infer/identity_shape_infer.h"
#include "graph/infer/shape_infer/broadcast_shape_infer.h"
#include "graph/infer/shape_infer/conv_shape_infer.h"

namespace my_inference {
    inline void inferShape(OpNode *op) {
        static std::map<OpType, ShapeInfer *> map{
            // activate
            {OpType::Relu, IdentityShapeInfer::instance()},
            {OpType::Sigmoid, IdentityShapeInfer::instance()},
            {OpType::Tanh, IdentityShapeInfer::instance()},
            {OpType::Gelu, IdentityShapeInfer::instance()},
            {OpType::Softmax, IdentityShapeInfer::instance()},
            {OpType::LeakyRelu, IdentityShapeInfer::instance()},
            // one math
            {OpType::Exp, IdentityShapeInfer::instance()},
            {OpType::Log, IdentityShapeInfer::instance()},
            {OpType::Sqrt, IdentityShapeInfer::instance()},
            {OpType::Abs, IdentityShapeInfer::instance()},
            {OpType::Neg, IdentityShapeInfer::instance()},
            {OpType::Erf, IdentityShapeInfer::instance()},
            {OpType::Not, IdentityShapeInfer::instance()},
            // convert
            {OpType::Identity, IdentityShapeInfer::instance()},
            {OpType::Cast, IdentityShapeInfer::instance()},
            {OpType::Clip, IdentityShapeInfer::instance()},
            // normalize
            {OpType::LayerNormalization, IdentityShapeInfer::instance()},
            {OpType::BatchNormalization, IdentityShapeInfer::instance()},
            {OpType::InstanceNormalization, IdentityShapeInfer::instance()},
            // two math
            {OpType::Add, BroadcastShapeInfer::instance()},
            {OpType::Sub, BroadcastShapeInfer::instance()},
            {OpType::Mul, BroadcastShapeInfer::instance()},
            {OpType::Div, BroadcastShapeInfer::instance()},
            {OpType::Pow, BroadcastShapeInfer::instance()},
            // logic
            {OpType::Less, BroadcastShapeInfer::instance()},
            {OpType::Greater, BroadcastShapeInfer::instance()},
            {OpType::Equal, BroadcastShapeInfer::instance()},
            {OpType::And, BroadcastShapeInfer::instance()},
            {OpType::Or, BroadcastShapeInfer::instance()},
            {OpType::Where, BroadcastShapeInfer::instance()},
            // conv
            {OpType::Conv, ConvShapeInfer::instance()},

        };
        const auto it = map.find(op->type());
        if (it == map.end()) {
            return;
        }
        (*it->second)(op);
    }
}
#endif //MY_INFERENCE_SHAPE_INFER_UTIL_H
