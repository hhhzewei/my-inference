//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_TRANSPOSE_KERNEL_H
#define MY_INFERENCE_TRANSPOSE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/generic/transpose.h"

namespace my_inference::cpu::generic {
    template<typename T>
    class TransposeKernel : public OpKernel {
    public:
        explicit TransposeKernel(const OpNode *op): OpKernel(op) {
            const int num_dim = op->input(0)->numDim();
            args_.num_dim = num_dim;
            args_.num_elem = op->input(0)->numData().value();
            const auto perm = op->attribute<std::vector<int64_t> >(AttributeKey::Perm).value();
            for (int i = 0; i < num_dim; ++i) {
                args_.x_shape[i] = op->input(0)->dim(i).value();
                args_.y_shape[i] = op->output(0)->dim(i).value();
                args_.x_strides[i] = op->inputStrides(0)[i].value();
                args_.y_strides[i] = op->outputStrides(0)[i].value();
                args_.perm[i] = perm[i];
            }
        }

        void operator()(const KernelParam &ctx) override {
            primitive::transpose(
                static_cast<T *>(ctx.inputs[0]),
                static_cast<T *>(ctx.outputs[0]),
                args_);
        }

    private:
        TransposeArgs args_{};
    };
}
#endif //MY_INFERENCE_TRANSPOSE_KERNEL_H
