//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#define MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#include "kernel/cpu/element_wise_kernel.h"
#include "kernel/kernel_creator/kernel_creator.h"
#include "util/singleton.h"

namespace my_inference {
    template<typename Func, typename T, DeviceType DEVICE_TYPE>
    class BinaryElementWiseKernelWithStrideCreator :
            public KernelCreator,
            public Singleton<BinaryElementWiseKernelWithStrideCreator<Func, T, DEVICE_TYPE> > {
    public:
        std::unique_ptr<OpKernel> operator()(OpNode *op) override {
            const int numDim = op->output(0)->numDim();
            if (numDim == 1) {
                if constexpr (DEVICE_TYPE == DeviceType::CPU) {
                    return std::make_unique<cpu::BinaryElementWiseWithStridesKernel1D<T, Func> >(op);
                }
            } else if (numDim == 2) {
                if constexpr (DEVICE_TYPE == DeviceType::CPU) {
                    return std::make_unique<cpu::BinaryElementWiseWithStridesKernel2D<T, Func> >(op);
                }
            } else {
                if constexpr (DEVICE_TYPE == DeviceType::CPU) {
                    return std::make_unique<cpu::BinaryElementWiseWithStridesKernelND<T, Func> >(op);
                }
            }
            return {};
        }
    };
}
#endif //MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
