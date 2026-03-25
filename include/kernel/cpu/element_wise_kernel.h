//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_KERNEL_H
#define MY_INFERENCE_ELEMENT_WISE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/element_wise.h"

namespace my_inference::cpu {
    template<typename T, typename Func>
    class BinaryElementWiseKernel : public OpKernel {
    public:
        explicit BinaryElementWiseKernel(const OpNode *op) : N(op->output(0)->numData().value()) {
        }

        void operator()(const KernelContext &ctx) override {
            primitive::binaryElementWise<T, Func>(
                static_cast<T *>(ctx.inputs[0]), static_cast<T *>(ctx.inputs[1]),
                static_cast<T *>(ctx.outputs[0]),
                N);
        }

    private:
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementWiseWithStridesKernel1D : public OpKernel {
    public:
        explicit BinaryElementWiseWithStridesKernel1D(const OpNode *op) : N(op->output(0)->dim(0).value()) {
        }

        void operator()(const KernelContext &ctx) override {
            primitive::binaryElementWiseWithStrides1D<T, Func>(
                static_cast<T *>(ctx.inputs[0]), static_cast<int64_t *>(ctx.inputs[1]),
                static_cast<T *>(ctx.inputs[2]), static_cast<int64_t *>(ctx.inputs[3]),
                static_cast<T *>(ctx.outputs[0]),
                N);
        }

    private:
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementWiseWithStridesKernel2D : public OpKernel {
    public:
        explicit BinaryElementWiseWithStridesKernel2D(const OpNode *op) : M(op->output(0)->dim(0).value()),
                                                                          N(op->output(0)->dim(1).value()) {
        }

        void operator()(const KernelContext &ctx) override {
            primitive::binaryElementWiseWithStrides2D<T, Func>(
                static_cast<T *>(ctx.inputs[0]), static_cast<int64_t *>(ctx.inputs[1]),
                static_cast<T *>(ctx.inputs[2]), static_cast<int64_t *>(ctx.inputs[3]),
                static_cast<T *>(ctx.outputs[0]),
                M, N);
        }

    private:
        int64_t M;
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementWiseWithStridesKernelND : public OpKernel {
    public:
        explicit BinaryElementWiseWithStridesKernelND(const OpNode *op) : num_data(op->output(0)->numData().value()),
                                                                          num_dim(op->output(0)->numDim()) {
        }

        void operator()(const KernelContext &ctx) override {
            primitive::binaryElementWiseWithStridesND<T, Func>(
                static_cast<T *>(ctx.inputs[0]), static_cast<int64_t *>(ctx.inputs[1]),
                static_cast<T *>(ctx.inputs[2]), static_cast<int64_t *>(ctx.inputs[3]),
                static_cast<T *>(ctx.outputs[0]), static_cast<int64_t *>(ctx.inputs[1]),
                static_cast<int64_t *>(ctx.inputs[4]), num_data, num_dim);
        }

    private:
        int64_t num_data;
        int64_t num_dim;
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_KERNEL_H
