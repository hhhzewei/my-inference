//
// Created by hzw on 2026/2/23.
//
#pragma once
#include "util/util.h"
#include "kernel/primitive/cpu/element_wise.h"
#include "optimize/constant_folder/op_folder.h"
#include "graph/node/tensor_node.h"

namespace my_inference {
    template<typename T, typename Func>
    class ElementWiseFolder : public OpFolder, public Singleton<ElementWiseFolder<T, Func> > {
        DECLARE_SINGLETON(ElementWiseFolder)

    public:
        void operator()(OpNode *op) override {
            const T *a = static_cast<T *>(op->input(0)->data());
            const T *b = static_cast<T *>(op->input(1)->data());
            const std::vector<int64_t> a_strides = toValue(op->inputStrides(0));
            const std::vector<int64_t> b_strides = toValue(op->inputStrides(1));
            const std::vector<int64_t> c_strides = toValue(op->outputStrides(0));
            const std::vector<int64_t> shape = toValue(op->output(0)->shape());
            int64_t numElem = 1;
            for (const int64_t &dim: shape) {
                numElem *= dim;
            }
            // 创建输出内存
            T *c = static_cast<T *>(malloc(numElem * sizeof(T)));
            op->output(0)->setData(c);
            // 调用原语
            if (shape.size() == 1) {
                kernel::primitive::cpu::elementWiseWithStrides1D<T, Func>(
                    a, a_strides.data(),
                    b, b_strides.data(),
                    c,
                    numElem);
            } else if (shape.size() == 2) {
                kernel::primitive::cpu::elementWiseWithStrides2D<T, Func>(
                    a, a_strides.data(),
                    b, b_strides.data(),
                    c, shape.data(), c_strides.data());
            }
        }
    };
}
