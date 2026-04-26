//
// Created by hzw on 2026/4/24.
//
#pragma once

#include "kernel/cpu/avx512/conv/conv2D_base_kernel.h"
#include "kernel/primitive/cpu/avx512/conv/standard_conv2D_kernel.h"

namespace my_inference::cpu::avx512 {
    template<typename T>
    class StandardConv2DKernel : public Conv2DBaseKernel {
    public:
        explicit StandardConv2DKernel(const OpNode *op) : Conv2DBaseKernel(op) {
        }

        void operator()(const KernelParam &ctx) override {
            void *bias = ctx.inputs.size() == 2 ? nullptr : ctx.inputs[2];
            primitive::standard_conv2D(static_cast<T *>(ctx.inputs[0]), static_cast<T *>(ctx.inputs[1]),
                                       static_cast<T *>(bias), static_cast<T *>(ctx.outputs[0]),
                                       args_);
        }
    };
}
