//
// Created by hzw on 2026/2/23.
//
#pragma once
#include "graph/node/tensor_node.h"
#include "../../kernel/primitive/cpu/generic/element_wise.h"
#include "optimize/constant_folder/op_folder.h"
#include "util/singleton.h"

namespace my_inference {
    template<typename T, BinaryOpType op_type>
    class ElementwiseFolder : public OpFolder, public Singleton<ElementwiseFolder<T, op_type> > {
        DECLARE_SINGLETON(ElementwiseFolder)

    public:
        void operator()(OpNode *op) override {
            const T *a = static_cast<T *>(op->input(0)->data());
            const T *b = static_cast<T *>(op->input(1)->data());
            const auto c_tensor = op->output(0);
            const int64_t num_elem = c_tensor->numData().value();
            // 创建输出内存
            T *c = static_cast<T *>(malloc(num_elem * sizeof(T)));
            op->output(0)->setData(c);
            // 调用原语
            const auto num_dim = static_cast<int64_t>(c_tensor->numDim());
            BroadcastBinaryElementwiseArgs args{};
            args.num_elem = num_elem;
            args.num_dim = num_dim;
            for (int i = 0; i < num_dim; ++i) {
                args.shape[i] = c_tensor->dim(i).value();
                args.a_strides[i] = op->inputStrides(0)[i].value();
                args.b_strides[i] = op->inputStrides(1)[i].value();
            }
            cpu::generic::primitive::broadcastBinaryElementWiseND<T, op_type>(a, b, c, args);
        }
    };
}
