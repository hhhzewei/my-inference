//
// Created by hzw on 2026/4/26.
//
#pragma once

#include "util/arithmetic_op_type/binary_op_type.h"
#include "util/arithmetic_op_type/unary_op_type.h"

namespace my_inference::cpu::generic {
    template<typename T>
    struct Traits {
        static T relu6(T x) {
            return x < 0 ? 0 : x > 6 ? 6 : x;
        }

        static T add(T a, T b) {
            return a + b;
        }

        static T sub(T a, T b) {
            return a - b;
        }

        static T mul(T a, T b) {
            return a * b;
        }

        static T div(T a, T b) {
            return a / b;
        }

        template<UnaryOpType op_type>
        static T unaryOp(T x) {
            if constexpr (op_type == UnaryOpType::Relu6) {
                return Traits::relu6(x);
            }
            return 0;
        }

        template<BinaryOpType op_type>
        static T binaryOp(T a, T b) {
            if constexpr (op_type == BinaryOpType::Add) {
                return Traits::add(a, b);
            }
            if constexpr (op_type == BinaryOpType::Sub) {
                return Traits::sub(a, b);
            }
            if constexpr (op_type == BinaryOpType::Mul) {
                return Traits::mul(a, b);
            }
            if constexpr (op_type == BinaryOpType::Div) {
                return Traits::div(a, b);
            }
            return 0;
        }
    };
}
