//
// Created by hzw on 2026/2/17.
//
#pragma once
#include <vector>

namespace my_inference {
    struct KernelContext {
        std::vector<void *> inputs;
        std::vector<void *> outputs;
    };

    class OpKernel {
    public:
        virtual ~OpKernel() = default;

        virtual void operator()(const KernelContext &ctx) = 0;
    };
}
