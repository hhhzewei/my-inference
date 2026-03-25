//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#define MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
#include "kernel/kernel_creator.h"
#include "kernel/kernel_util.h"
#include "kernel/cpu/element_wise_kernel.h"
#include "kernel/kernel_key_generator/element_wise_kernel_key_generator.h"
#include "util/math.h"

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
        }
    };

    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Add,DeviceType::CPU,DataType::Float32,false),
        &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, AddFunctor<float> > >::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Sub,DeviceType::CPU,DataType::Float32,false),
        &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, SubFunctor<float> > >::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Mul,DeviceType::CPU,DataType::Float32,false),
        &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, MulFunctor<float> > >::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Div,DeviceType::CPU,DataType::Float32,false),
        &(GenericKernelCreator<cpu::BinaryElementWiseKernel<float, DivFunctor<float> > >::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Add,DeviceType::CPU,DataType::Float32,true),
        &(BinaryElementWiseKernelWithStrideCreator<AddFunctor<float>,float,DeviceType::CPU>::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Sub,DeviceType::CPU,DataType::Float32,true),
        &(BinaryElementWiseKernelWithStrideCreator<SubFunctor<float>,float,DeviceType::CPU>::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Mul,DeviceType::CPU,DataType::Float32,true),
        &(BinaryElementWiseKernelWithStrideCreator<MulFunctor<float>,float,DeviceType::CPU>::instance()));
    REGISTER_KERNEL_CREATOR(
        ElementWiseKeyGenerator::generate(OpType::Div,DeviceType::CPU,DataType::Float32,true),
        &(BinaryElementWiseKernelWithStrideCreator<DivFunctor<float>,float,DeviceType::CPU>::instance()));
}
#endif //MY_INFERENCE_ELEMENT_WISE_WITH_STRIDE_KERNEL_CREATOR_H
