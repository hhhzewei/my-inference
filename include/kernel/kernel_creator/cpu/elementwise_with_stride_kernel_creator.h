//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#define MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#include "kernel/cpu/generic/elementwise/broadcast_binary_elementwise1D_kernel.h"
#include "kernel/cpu/generic/elementwise/broadcast_binary_elementwise2D_kernel.h"
#include "kernel/cpu/generic/elementwise/broadcast_binary_elementwiseND_kernel.h"
#include "kernel/kernel_creator/kernel_creator.h"
#include "util/singleton.h"

namespace my_inference::cpu::generic {
    template<typename T, BinaryOpType op_type>
    class BinaryElementwiseKernelWithStrideCreator :
            public KernelCreator,
            public Singleton<BinaryElementwiseKernelWithStrideCreator<T, op_type> > {
    public:
        std::unique_ptr<OpKernel> operator()(OpNode *op) override {
            const int numDim = op->output(0)->numDim();
            if (numDim == 1) {
                return std::make_unique<BroadcastBinaryElementwise1DKernel<T, op_type> >(op);
            }
            if (numDim == 2) {
                return std::make_unique<BroadcastBinaryElementwise2DKernel<T, op_type> >(op);
            }
            return std::make_unique<BroadcastBinaryElementwiseNDKernel<T, op_type> >(op);
        }
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
