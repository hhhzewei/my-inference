//
// Created by hzw on 2026/4/26.
//
#pragma once
#include "backend/isa_traits/avx512_traits.h"
#include "kernel/kernel_args/elementwise_args.h"
#include "util/arithmetic_op_type/unary_op_type.h"

namespace my_inference::cpu::avx512::primitive {
    template<typename T, UnaryOpType op_type>
    void unaryElementWise(
        // 输入数据指针
        const T *x,
        // 输出数据指针 (Output)
        T *y,
        const ElementWiseArgs args) {
        using traits = Traits<T>;
        for (int64_t i = 0; i < args.num_elem; i += traits::NumPerVec) {
            auto vec = traits::load(x + i);
            vec = traits::unaryOp < op_type > (vec);
            traits::store(y + i, vec);
        }
    }

    template<typename T, BinaryOpType op_type>
    void binaryElementWise(
        // 输入数据指针
        const T *x1, const T *x2,
        // 输出数据指针 (Output)
        T *y,
        const ElementWiseArgs args) {
        using traits = Traits<T>;
        for (int64_t i = 0; i < args.num_elem; i += traits::NumPerVec) {
            auto vec1 = traits::load(x1 + i);
            auto vec2 = traits::load(x2 + i);
            vec1 = traits::binaryOp < op_type > (vec1, vec2);
            traits::store(y + i, vec1);
        }
    }
}
