//
// Created by hzw on 2026/4/2.
//
#pragma once

#include "graph/node/op_node.h"
#include "kernel/cpu/generic/conv/conv2D_base_kernel.h"
#include "kernel/primitive/cpu/generic/conv2D/grouped_conv2D.h"

namespace my_inference::cpu::generic {
    template<typename T>
    class GroupedConv2DKernel : public Conv2DBaseKernel {
    public:
        explicit GroupedConv2DKernel(const OpNode *op) : Conv2DBaseKernel(op) {
        }

        void operator()(const KernelParam &ctx) override {
            void *bias = ctx.inputs.size() == 2 ? nullptr : ctx.inputs[2];
            primitive::grouped_conv2D(static_cast<T *>(ctx.inputs[0]), static_cast<T *>(ctx.inputs[1]),
                                      static_cast<T *>(bias), static_cast<T *>(ctx.outputs[0]),
                                      args_);
        }
    };
}
