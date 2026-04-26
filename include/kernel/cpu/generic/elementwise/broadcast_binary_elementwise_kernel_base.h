//
// Created by hzw on 2026/4/26.
//
#pragma once
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_args/broadcast_binary_elementwise_args.h"

namespace my_inference::cpu::generic {
    class BroadcastBinaryElementwiseKernelBase : public OpKernel {
    public:
        explicit BroadcastBinaryElementwiseKernelBase(const OpNode *op) : OpKernel(op) {
            const TensorNode *output = op->output(0);
            args_.num_elem = output->numData().value();
            args_.num_dim = output->numDim();
            for (int i = 0; i < args_.num_dim; ++i) {
                args_.shape[i] = output->dim(i).value();
            }
            for (int i = 0; i < args_.num_dim; ++i) {
                args_.a_strides[i] = op->inputStrides(0)[i].value();
            }
            for (int i = 0; i < args_.num_dim; ++i) {
                args_.b_strides[i] = op->inputStrides(1)[i].value();
            }
        }

    protected:
        BroadcastBinaryElementwiseArgs args_{};
    };
}
