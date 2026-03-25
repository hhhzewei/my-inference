//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_SHAPE_INFER_UTIL_H
#define MY_INFERENCE_SHAPE_INFER_UTIL_H
#include "graph/shape_infer/shape_infer.h"
#include "util/factory.h"

namespace my_inference {
#define REGISTER_SHAPE_INFER(op_type,shape_infer) GENERIC_REGISTER(OpType,ShapeInfer*,op_type,shape_infer)

    inline void inferShape(OpNode *op) {
        using ShapeInferFactory = GenericFactory<OpType, ShapeInfer *>;
        if (ShapeInfer *shape_infer = ShapeInferFactory::instance().get(op->type())) {
            (*shape_infer)(op);
        }
    }
}
#endif //MY_INFERENCE_SHAPE_INFER_UTIL_H
