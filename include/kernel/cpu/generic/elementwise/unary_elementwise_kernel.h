//
// Created by hzw on 2026/4/26.
//
#pragma once
#include "kernel/cpu/generic/elementwise/elementwise_kernel_base.h"
#include "kernel/primitive/cpu/generic/element_wise.h"

namespace my_inference::cpu::generic {
    template<typename T, UnaryOpType op_type>
    class UnaryElementwiseKernel : public ElementwiseKernelBase {
    public:
        explicit UnaryElementwiseKernel(const OpNode *op) : ElementwiseKernelBase(op) {
        }

        void operator()(const KernelParam &param) override {
            primitive::unaryElementWise<T, op_type>(
                static_cast<T *>(param.inputs[0]),
                static_cast<T *>(param.outputs[0]),
                args_);
        }
    };
}
