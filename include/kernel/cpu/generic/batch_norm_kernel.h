//
// Created by hzw on 2026/4/1.
//

#ifndef MY_INFERENCE_BATCH_NORM_KERNEL_H
#define MY_INFERENCE_BATCH_NORM_KERNEL_H
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_args/batch_norm_args.h"
#include "kernel/primitive/cpu/generic/batch_norm.h"

namespace my_inference::cpu::generic {
    template<typename T>
    class BatchNormKernel : public OpKernel {
    public:
        explicit BatchNormKernel(const OpNode *op) : OpKernel(op) {
            args_.N = op->input(0)->dim(0).value();
            args_.C = op->input(0)->dim(1).value();
            args_.stride = op->inputStrides(0)[1].value();
            args_.eps = op->attribute<float>(AttributeKey::Epsilon).value();
        }

        void operator()(const KernelParam &ctx) override {
            primitive::batchNormalize(static_cast<T *>(ctx.inputs[0]),
                                      static_cast<float *>(ctx.inputs[1]),
                                      static_cast<float *>(ctx.inputs[2]),
                                      static_cast<T *>(ctx.inputs[3]), static_cast<T *>(ctx.inputs[4]),
                                      static_cast<T *>(ctx.outputs[0]),
                                      args_);
        }

    private:
        BatchNormalizeArgs args_{};
    };
}
#endif //MY_INFERENCE_BATCH_NORM_KERNEL_H
