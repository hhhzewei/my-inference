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
    class unaryElementwiseKernel : public OpKernel {
    public:
        explicit unaryElementwiseKernel(const OpNode *op) : N(op->output(0)->numData().value()) {
        }

        void operator()(const KernelParam &param) override {
            primitive::unaryElementWise<T, Func>(
                static_cast<T *>(param.inputs[0].tensor),
                static_cast<T *>(param.outputs[0].tensor),
                N);
        }

    private:
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementwiseKernel : public OpKernel {
    public:
        explicit BinaryElementwiseKernel(const OpNode *op) : N(op->output(0)->numData().value()) {
        }

        void operator()(const KernelParam &param) override {
            primitive::binaryElementWise<T, Func>(
                static_cast<T *>(param.inputs[0].tensor), static_cast<T *>(param.inputs[1].tensor),
                static_cast<T *>(param.outputs[0].tensor),
                N);
        }

    private:
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementwiseWithStridesKernel1D : public OpKernel {
    public:
        explicit BinaryElementwiseWithStridesKernel1D(const OpNode *op) : N(op->output(0)->dim(0).value()) {
        }

        void operator()(const KernelParam &param) override {
            primitive::binaryElementWiseWithStrides1D<T, Func>(
                static_cast<T *>(param.inputs[0].tensor), param.inputs[0].strides,
                static_cast<T *>(param.inputs[1].tensor), param.inputs[1].strides,
                static_cast<T *>(param.outputs[0].tensor),
                N);
        }

    private:
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementwiseWithStridesKernel2D : public OpKernel {
    public:
        explicit BinaryElementwiseWithStridesKernel2D(const OpNode *op) : M(op->output(0)->dim(0).value()),
                                                                          N(op->output(0)->dim(1).value()) {
        }

        void operator()(const KernelParam &param) override {
            primitive::binaryElementwiseWithStrides2D<T, Func>(
                static_cast<T *>(param.inputs[0].tensor), param.inputs[0].strides,
                static_cast<T *>(param.inputs[1].tensor), param.inputs[1].strides,
                static_cast<T *>(param.outputs[0].tensor),
                M, N);
        }

    private:
        int64_t M;
        int64_t N;
    };

    template<typename T, typename Func>
    class BinaryElementwiseWithStridesKernelND : public OpKernel {
    public:
        explicit BinaryElementwiseWithStridesKernelND(const OpNode *op) : num_data(op->output(0)->numData().value()),
                                                                          num_dim(op->output(0)->numDim()) {
        }

        void operator()(const KernelParam &param) override {
            primitive::binaryElementWiseWithStridesND<T, Func>(
                static_cast<T *>(param.inputs[0].tensor), param.inputs[0].strides,
                static_cast<T *>(param.inputs[1].tensor), param.inputs[1].strides,
                static_cast<T *>(param.outputs[0].tensor), param.outputs[0].strides,
                param.outputs[0].shape, num_data, num_dim);
        }

    private:
        int64_t num_data;
        int64_t num_dim;
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_KERNEL_H
