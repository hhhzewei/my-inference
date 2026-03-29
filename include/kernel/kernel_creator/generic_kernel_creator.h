//
// Created by hzw on 2026/3/29.
//

#ifndef MY_INFERENCE_GENERIC_KERNEL_CREATOR_H
#define MY_INFERENCE_GENERIC_KERNEL_CREATOR_H
#include "kernel/kernel_creator/kernel_creator.h"
#include "util/singleton.h"

namespace my_inference {
    template<typename T>
    class GenericKernelCreator : public KernelCreator, public Singleton<GenericKernelCreator<T> > {
        DECLARE_SINGLETON(GenericKernelCreator)

    public:
        std::unique_ptr<OpKernel> operator()(OpNode *op) override {
            return std::make_unique<T>(op);
        }
    };
}

#endif //MY_INFERENCE_GENERIC_KERNEL_CREATOR_H
