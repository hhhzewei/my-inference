//
// Created by hzw on 2026/3/24.
//

#ifndef MY_INFERENCE_KERNEL_CREATOR_H
#define MY_INFERENCE_KERNEL_CREATOR_H
#include "graph/node/op_node.h"
#include "kernel/op_kernel.h"

namespace my_inference {
    class KernelCreator {
    public:
        virtual ~KernelCreator() = default;

        virtual std::unique_ptr<OpKernel> operator()(OpNode *) =0;
    };
}

#endif //MY_INFERENCE_KERNEL_CREATOR_H
