//
// Created by hzw on 2026/2/17.
//
#pragma once
#include "kernel_key.h"
#include "graph/op_node.h"

namespace my_inference {
    class OpKernel {
    public:
        virtual ~OpKernel() = default;

        virtual void operator()(void *rt_ctx, void *static_attr) = 0;
    };

    inline OpKernel *getOpKernel(const OpNode &op_node) {
        static const std::unordered_map<KernelKey, OpKernel *> map = {};
        const KernelKey key = getKernelKey(op_node);
        const auto it = map.find(key);
        if (it == map.end()) {
            std::cout << "Cant find kernel";
            return nullptr;
        }
        OpKernel *kernel = it->second;
        return kernel;
    }
}
