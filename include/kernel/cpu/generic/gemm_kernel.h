//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_GEMM_KERNEL_H
#define MY_INFERENCE_GEMM_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/generic/gemm.h"

namespace my_inference::cpu::generic {
    template<typename T, bool TransA, bool TransB>
    class GemmKernel : public OpKernel {
    public:
        explicit GemmKernel(const OpNode *op): OpKernel(op) {
            if constexpr (!TransA) {
                args_.K = op->input(0)->dim(1).value();
            } else {
                args_.K = op->input(0)->dim(0).value();
            }
            args_.M = op->output(0)->dim(0).value();
            args_.N = op->output(0)->dim(1).value();
            args_.alpha = op->attribute<float>(AttributeKey::Alpha).value();
            args_.beta = op->attribute<float>(AttributeKey::Beta).value();
            args_.bias_stride[0] = op->inputStrides(2)[0].value();
            args_.bias_stride[1] = op->inputStrides(2)[1].value();
        }

        void operator()(const KernelParam &ctx) override {
            primitive::gemm<T, TransA, TransB>(static_cast<T *>(ctx.inputs[0]), static_cast<T *>(ctx.inputs[1]),
                                               static_cast<T *>(ctx.inputs[2]),
                                               static_cast<T *>(ctx.outputs[0]),
                                               args_);
        }

    private:
        primitive::GemmArgs args_;
    };
}
#endif //MY_INFERENCE_GEMM_KERNEL_H
