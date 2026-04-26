//
// Created by hzw on 2026/2/17.
//
#pragma once

namespace my_inference {
    struct KernelParam {
        std::vector<void *> inputs;
        std::vector<void *> outputs;
    };

    class OpKernel {
    public:
        explicit OpKernel(const OpNode *op) : op_(op) {
        }

        [[nodiscard]] const OpNode *op() const {
            return op_;
        }

        virtual ~OpKernel() = default;

        virtual void operator()(const KernelParam &ctx) = 0;

    private:
        const OpNode *op_;
    };
}
