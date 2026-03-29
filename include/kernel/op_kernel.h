//
// Created by hzw on 2026/2/17.
//
#pragma once
#include <vector>

namespace my_inference {
    struct KernelParam {
        struct TensorDesc {
            void *tensor;
            int64_t *shape;
            int64_t *strides;
        };

        std::vector<TensorDesc> inputs;
        std::vector<TensorDesc> outputs;
    };

    class OpKernel {
    public:
        virtual ~OpKernel() = default;

        virtual void operator()(const KernelParam &ctx) = 0;
    };
}
