//
// Created by hzw on 2026/2/23.
//
#pragma once
#include "graph/node/tensor_node.h"
#include "kernel/primitive/cpu/element_wise.h"
#include "optimize/constant_folder/op_folder_util.h"
#include "util/math.h"

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
                cpu::primitive::binaryElementWiseWithStrides1D<T, Func>(
                    a, a_strides.data(),
                    b, b_strides.data(),
                    c,
                    numElem);
            } else if (shape.size() == 2) {
                cpu::primitive::binaryElementWiseWithStrides2D<T, Func>(
                    a, a_strides.data(),
                    b, b_strides.data(),
                    c, shape[0], shape[1]);
            }
        }
    };

    REGISTER_OP_FOLDER(getFolderKey(OpType::Add, DataType::Float32, DeviceType::CPU),
                       (&ElementWiseFolder<float, AddFunctor<float> >::instance()));
    REGISTER_OP_FOLDER(getFolderKey(OpType::Sub, DataType::Float32, DeviceType::CPU),
                       (&ElementWiseFolder<float, SubFunctor<float> >::instance()));
    REGISTER_OP_FOLDER(getFolderKey(OpType::Mul, DataType::Float32, DeviceType::CPU),
                       (&ElementWiseFolder<float, MulFunctor<float> >::instance()));
    REGISTER_OP_FOLDER(getFolderKey(OpType::Div, DataType::Float32, DeviceType::CPU),
                       (&ElementWiseFolder<float, DivFunctor<float> >::instance()));
}
