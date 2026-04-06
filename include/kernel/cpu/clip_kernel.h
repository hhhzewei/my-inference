//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CLIP_KERNEL_H
#define MY_INFERENCE_CLIP_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/clip.h"

namespace my_inference::cpu {
    template<typename T>
    class ClipKernel : public OpKernel {
    public:
        explicit ClipKernel(const OpNode *op) : N(op->input(0)->numData().value()) {
        }

        void operator()(const KernelParam &ctx) override {
            primitive::clip(static_cast<T *>(ctx.inputs[0].tensor),
                            static_cast<T *>(ctx.inputs[1].tensor), static_cast<T *>(ctx.inputs[2].tensor),
                            static_cast<T *>(ctx.outputs[0].tensor),
                            N);
        }

    private:
        int64_t N;
    };
}
#endif //MY_INFERENCE_CLIP_KERNEL_H
