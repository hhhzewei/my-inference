//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#define MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#include "kernel/cpu/element_wise_kernel.h"
#include "kernel/kernel_creator/kernel_creator.h"
#include "util/singleton.h"

namespace my_inference::cpu {
    template<typename Func, typename T>
    class BinaryElementWiseKernelWithStrideCreator :
            public KernelCreator,
            public Singleton<BinaryElementWiseKernelWithStrideCreator<Func, T> > {
    public:
        std::unique_ptr<OpKernel> operator()(OpNode *op) override {
            const int numDim = op->output(0)->numDim();
            if (numDim == 1) {
                return std::make_unique<BinaryElementWiseWithStridesKernel1D<T, Func> >(op);
            }
            if (numDim == 2) {
                return std::make_unique<BinaryElementWiseWithStridesKernel2D<T, Func> >(op);
            }
            return std::make_unique<BinaryElementWiseWithStridesKernelND<T, Func> >(op);
        }
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
