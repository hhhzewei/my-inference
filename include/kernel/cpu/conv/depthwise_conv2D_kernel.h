//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CONV2D_DEPTHWISE_KERNEL_H
#define MY_INFERENCE_CONV2D_DEPTHWISE_KERNEL_H
#include "graph/node/op_node.h"
#include "kernel/cpu/conv/conv2D_base_kernel.h"
#include "kernel/primitive/cpu/conv2D/depthwise_conv2D.h"

namespace my_inference::cpu {
    template<typename T>
    class DepthwiseConv2dKernel : public Conv2DBaseKernel {
    public:
        explicit DepthwiseConv2dKernel(const OpNode *op) : Conv2DBaseKernel(op) {
        }

        void operator()(const KernelParam &ctx) override {
            void *bias = ctx.inputs.size() == 2 ? nullptr : ctx.inputs[2].tensor;
            primitive::depthwise_conv2D(static_cast<T *>(ctx.inputs[0].tensor), static_cast<T *>(ctx.inputs[1].tensor),
                                        static_cast<T *>(ctx.inputs[2].tensor), static_cast<T *>(bias),
                                        N, C_IN, H_IN, W_IN,
                                        C_OUT, H_OUT, W_OUT,
                                        K_H, K_W, PAD_UP, PAD_DOWN, PAD_LEFT, PAD_RIGHT,
                                        STRIDE_H, STRIDE_W,
                                        DILATION_H, DILATION_W);
        }
    };
}
#endif //MY_INFERENCE_CONV2D_DEPTHWISE_KERNEL_H
