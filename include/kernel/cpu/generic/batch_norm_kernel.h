//
// Created by hzw on 2026/4/1.
//

#ifndef MY_INFERENCE_BATCH_NORM_KERNEL_H
#define MY_INFERENCE_BATCH_NORM_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/generic/batch_norm.h"

namespace my_inference::cpu::generic {
    template<typename T>
    class BatchNormKernel : public OpKernel {
    public:
        explicit BatchNormKernel(const OpNode *op) {
            N = op->input(0)->dim(0).value();
            C = op->input(0)->dim(1).value();
            stride = op->inputStrides(0)[1].value();
            eps = op->attribute<float>(AttributeKey::Epsilon).value();
        }

        void operator()(const KernelParam &ctx) override {
            primitive::batchNorm(static_cast<T *>(ctx.inputs[0].tensor),
                                 static_cast<float *>(ctx.inputs[1].tensor), static_cast<float *>(ctx.inputs[2].tensor),
                                 static_cast<T *>(ctx.inputs[3].tensor), static_cast<T *>(ctx.inputs[4].tensor),
                                 static_cast<T *>(ctx.outputs[0].tensor),
                                 eps,
                                 N, C, stride);
        }

    private:
        int64_t N;
        int64_t C;
        int64_t stride;
        float eps;
    };
}
#endif //MY_INFERENCE_BATCH_NORM_KERNEL_H
