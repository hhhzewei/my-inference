//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_GEMM_KERNEL_H
#define MY_INFERENCE_GEMM_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/gemm.h"

namespace my_inference::cpu {
    template<typename T, bool TransA, bool TransB>
    class GemmKernel : public OpKernel {
    public:
        explicit GemmKernel(const OpNode *op) {
            if constexpr (!TransA) {
                K = op->input(0)->dim(1).value();
            } else {
                K = op->input(0)->dim(0).value();
            }
            M = op->output(0)->dim(0).value();
            N = op->output(0)->dim(1).value();
            alpha = op->attribute<float>(AttributeKey::Alpha).value();
            beta = op->attribute<float>(AttributeKey::Beta).value();
            bias_strides[0] = op->inputStrides(2)[0].value();
            bias_strides[1] = op->inputStrides(2)[1].value();
        }

        void operator()(const KernelParam &ctx) override {
            primitive::gemm<T, TransA, TransB>(static_cast<T *>(ctx.inputs[0].tensor),
                                               static_cast<T *>(ctx.inputs[1].tensor),
                                               static_cast<T *>(ctx.inputs[2].tensor),
                                               bias_strides[0], bias_strides[1],
                                               static_cast<T *>(ctx.outputs[0].tensor),
                                               M, K, N, alpha, beta);
        }

    private:
        int64_t M;
        int64_t K;
        int64_t N;
        float alpha;
        float beta;
        int64_t bias_strides[2]{1, 1};
    };
}
#endif //MY_INFERENCE_GEMM_KERNEL_H
