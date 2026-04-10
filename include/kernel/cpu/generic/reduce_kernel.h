//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_REDUCE_KERNEL_H
#define MY_INFERENCE_REDUCE_KERNEL_H
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "kernel/op_kernel.h"
#include "kernel/primitive/cpu/generic/reduce.h"
#include "util/math.h"

namespace my_inference::cpu::generic {
    template<typename T, typename ReducePolicy>
    class ReduceKernel : public OpKernel {
    public:
        explicit ReduceKernel(const OpNode *op) {
            std::vector<int64_t> axes = op->attribute<std::vector<int64_t> >(AttributeKey::Axes).value();
            auto &shape = op->input(0)->shape();
            int axe = 0;
            for (; axe < shape.size(); ++axe) {
                if (std::find(axes.begin(), axes.end(), axe) != axes.end()) {
                    break;
                }
                Outer *= shape[axe].value();
            }
            for (; axe < shape.size(); ++axe) {
                if (std::find(axes.begin(), axes.end(), axe) == axes.end()) {
                    break;
                }
                Reduce *= shape[axe].value();
            }
            for (; axe < shape.size(); ++axe) {
                Inner *= shape[axe].value();
            }
        }

        void operator()(const KernelParam &ctx) override {
            primitive::reduce<T, ReducePolicy>(
                static_cast<T *>(ctx.inputs[0].tensor), static_cast<T *>(ctx.outputs[0].tensor),
                Outer, Reduce, Inner
            );
        }

    private:
        int64_t Outer = 1;
        int64_t Reduce = 1;
        int64_t Inner = 1;
    };

    template<typename T>
    struct ReduceMaxPolicy {
        using BinaryFunctor = MaxFunctor<T>;
        using PostFunctor = IdentityFunctor<T>;
        constexpr static T InitValue = std::numeric_limits<T>::min();
    };

    template<typename T>
    using ReduceMaxKernel = ReduceKernel<T, ReduceMaxPolicy<T> >;

    template<typename T>
    struct ReduceMinPolicy {
        using BinaryFunctor = MinFunctor<T>;
        using PostFunctor = IdentityFunctor<T>;
        constexpr static T InitValue = std::numeric_limits<T>::max();
    };

    template<typename T>
    using ReduceMinKernel = ReduceKernel<T, ReduceMinPolicy<T> >;

    template<typename T>
    struct ReduceMeanPolicy {
        using BinaryFunctor = AddFunctor<T>;
        using PostFunctor = DivFunctor<T>;
        constexpr static T InitValue = 0;
    };

    template<typename T>
    using ReduceMeanKernel = ReduceKernel<T, ReduceMeanPolicy<T> >;


    template<typename T>
    struct ReduceSumPolicy {
        using BinaryFunctor = AddFunctor<T>;
        using PostFunctor = IdentityFunctor<T>;
        constexpr static T InitValue = 0;
    };

    template<typename T>
    using ReduceSumKernel = ReduceKernel<T, ReduceSumPolicy<T> >;
}
#endif //MY_INFERENCE_REDUCE_KERNEL_H
