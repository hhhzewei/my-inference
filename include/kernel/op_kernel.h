//
// Created by hzw on 2026/2/17.
//
#pragma once
#include "op_key.h"
#include "graph/op_node.h"

struct OpKernelContext {
    void **inputs;
    void **outputs;
    void *attributes;
};

class OpKernel {
public:
    virtual ~OpKernel() = default;

    virtual void run(OpKernelContext ctx) =0;
};

inline OpKernel *getOpKernel(const OpNode &op_node) {
    static const std::unordered_map<OpKey, OpKernel *> map = {};
    const OpKey key = getOpKey(op_node);
    const auto it = map.find(key);
    if (it == map.end()) {
        std::cout << "Cant find kernel";
    }
    return it->second;
}
