//
// Created by hzw on 2026/4/24.
//

#ifndef MY_INFERENCE_SLICE_KERNEL_H
#define MY_INFERENCE_SLICE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_args/slice_args.h"
#include "kernel/primitive/cpu/generic/slice.h"

namespace my_inference::cpu::generic {
    template<typename T>
    class SliceKernel : public OpKernel {
    public:
        explicit SliceKernel(const OpNode *op): OpKernel(op) {
            const auto *output = op->output(0);
            args_.num_dim = op->input(0)->numDim();
            args_.y_num_elem = output->numData().value();
            for (int64_t i = 0; i < args_.num_dim; ++i) {
                args_.x_stride[i] = op->inputStrides(0)[i].value();
                args_.y_shape[i] = output->dim(i).value();
            }
        }

        void operator()(const KernelParam &ctx) override {
            primitive::slice(static_cast<T *>(ctx.inputs[0]),
                             static_cast<int64_t *>(ctx.inputs[1]), static_cast<int64_t *>(ctx.inputs[2]),
                             static_cast<T *>(ctx.outputs[0]),
                             args_);
        }

    private:
        SliceArgs args_{};
    };
}
#endif //MY_INFERENCE_SLICE_KERNEL_H
