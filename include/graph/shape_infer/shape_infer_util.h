//
// Created by hzw on 2026/2/25.
//

#ifndef MY_INFERENCE_SHAPE_INFER_UTIL_H
#define MY_INFERENCE_SHAPE_INFER_UTIL_H
#include "graph/shape_infer/shape_infer.h"
#include "util/factory.h"

#define REGISTER_SHAPE_INFER(op_type,shape_infer) GENERIC_REGISTER(my_inference::OpType,my_inference::ShapeInfer*,op_type,shape_infer)

namespace my_inference {
    void inferShape(OpNode *op);
}
#endif //MY_INFERENCE_SHAPE_INFER_UTIL_H
