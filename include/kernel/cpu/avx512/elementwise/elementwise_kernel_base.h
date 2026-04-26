//
// Created by hzw on 2026/4/26.
//
#pragma once

#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_args/elementwise_args.h"

namespace my_inference::cpu::avx512 {
    class ElementwiseKernelBase : public OpKernel {
    public:
        explicit ElementwiseKernelBase(const OpNode *op) : OpKernel(op),
                                                           args_{op->output(0)->numData().value()} {
        }

    protected:
        ElementWiseArgs args_;
    };
}
