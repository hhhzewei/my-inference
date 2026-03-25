//
// Created by hzw on 2026/3/24.
//

#ifndef MY_INFERENCE_KERNEL_CREATOR_H
#define MY_INFERENCE_KERNEL_CREATOR_H
#include "graph/node/op_node.h"
#include "kernel/op_kernel.h"
#include "util/Singleton.h"

namespace my_inference {
    class KernelCreator {
    public:
        virtual ~KernelCreator() = default;

        virtual std::unique_ptr<OpKernel> operator()(OpNode *) =0;
    };

    template<typename T>
    class GenericKernelCreator : public KernelCreator, public Singleton<GenericKernelCreator<T> > {
        DECLARE_SINGLETON(GenericKernelCreator)

    public:
        std::unique_ptr<OpKernel> operator()(OpNode *op) override {
            return std::make_unique<T>(op);
        }
    };
}

#endif //MY_INFERENCE_KERNEL_CREATOR_H
