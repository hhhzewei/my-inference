//
// Created by hzw on 2026/3/24.
//

#ifndef MY_INFERENCE_KERNEL_UTIL_H
#define MY_INFERENCE_KERNEL_UTIL_H
#include <memory>

#include "graph/node/op_node.h"
#include "kernel/op_kernel.h"

namespace my_inference {
    std::unique_ptr<OpKernel> getOpKernel(OpNode *op);
}
#endif //MY_INFERENCE_KERNEL_UTIL_H
