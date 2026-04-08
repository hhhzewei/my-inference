//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CONV2D_BASE_KERNEL_H
#define MY_INFERENCE_CONV2D_BASE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"

namespace my_inference::cpu {
    class Conv2DBaseKernel : public OpKernel {
    protected:
        explicit Conv2DBaseKernel(const OpNode *op) {
            auto &input_shape = op->input(0)->shape();
            N = input_shape[0].value();
            C_IN = input_shape[1].value();
            H_IN = input_shape[2].value();
            W_IN = input_shape[3].value();
            auto &kernel_shape = op->input(1)->shape();
            C_OUT = kernel_shape[0].value();
            K_W = kernel_shape[2].value();
            K_H = kernel_shape[3].value();
            auto &output_shape = op->output(0)->shape();
            H_OUT = output_shape[2].value();
            W_OUT = output_shape[3].value();
            const auto pads = op->attribute<std::vector<int64_t> >(AttributeKey::Pads).value();
            PAD_UP = pads[0];
            PAD_LEFT = pads[1];
            PAD_DOWN = pads[2];
            PAD_RIGHT = pads[3];
            const auto strides = op->attribute<std::vector<int64_t> >(AttributeKey::Strides).value();
            STRIDE_H = strides[0];
            STRIDE_W = strides[1];
            const auto dilations = op->attribute<std::vector<int64_t> >(AttributeKey::Dilations).value();
            DILATION_H = dilations[0];
            DILATION_W = dilations[1];
            GROUP = op->attribute<int64_t>(AttributeKey::Group).value();
        }

        int64_t N;
        int64_t C_IN;
        int64_t H_IN;
        int64_t W_IN;
        int64_t C_OUT;
        int64_t H_OUT;
        int64_t W_OUT;
        int64_t K_H;
        int64_t K_W;
        int64_t PAD_UP;
        int64_t PAD_DOWN;
        int64_t PAD_LEFT;
        int64_t PAD_RIGHT;
        int64_t STRIDE_H;
        int64_t STRIDE_W;
        int64_t DILATION_H;
        int64_t DILATION_W;
        int64_t GROUP;
    };
}

#endif //MY_INFERENCE_CONV2D_BASE_KERNEL_H
