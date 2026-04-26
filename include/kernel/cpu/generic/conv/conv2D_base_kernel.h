//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CONV2D_BASE_KERNEL_H
#define MY_INFERENCE_CONV2D_BASE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_args/conv2D_args.h"

namespace my_inference::cpu::generic {
    class Conv2DBaseKernel : public OpKernel {
    protected:
        explicit Conv2DBaseKernel(const OpNode *op): OpKernel(op) {
            auto &input_shape = op->input(0)->shape();
            args_.N = input_shape[0].value();
            args_.C_IN = input_shape[1].value();
            args_.H_IN = input_shape[2].value();
            args_.W_IN = input_shape[3].value();
            auto &kernel_shape = op->input(1)->shape();
            args_.C_OUT = kernel_shape[0].value();
            args_.W_K = kernel_shape[2].value();
            args_.H_K = kernel_shape[3].value();
            auto &output_shape = op->output(0)->shape();
            args_.H_OUT = output_shape[2].value();
            args_.W_OUT = output_shape[3].value();
            const auto pads = op->attribute<std::vector<int64_t> >(AttributeKey::Pads).value();
            args_.PAD_UP = pads[0];
            args_.PAD_LEFT = pads[1];
            args_.PAD_DOWN = pads[2];
            args_.PAD_RIGHT = pads[3];
            const auto strides = op->attribute<std::vector<int64_t> >(AttributeKey::Strides).value();
            args_.STRIDE_H = strides[0];
            args_.STRIDE_W = strides[1];
            const auto dilations = op->attribute<std::vector<int64_t> >(AttributeKey::Dilations).value();
            args_.DILATION_H = dilations[0];
            args_.DILATION_W = dilations[1];
            args_.GROUP = op->attribute<int64_t>(AttributeKey::Group).value();
        }

        Conv2DArgs args_{};
    };
}

#endif //MY_INFERENCE_CONV2D_BASE_KERNEL_H
