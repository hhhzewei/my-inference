//
// Created by hzw on 2026/3/29.
//

#include "graph/shape_infer/shape_infer_util.h"

void my_inference::inferShape(OpNode *op) {
    using ShapeInferFactory = GenericFactory<OpType, ShapeInfer *>;
    if (ShapeInfer *shape_infer = ShapeInferFactory::instance().get(op->type())) {
        (*shape_infer)(op);
    }
}
