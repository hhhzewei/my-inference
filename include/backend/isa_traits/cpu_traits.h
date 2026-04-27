//
// Created by hzw on 2026/4/26.
//
#pragma once

#include "util/arithmetic_op_type/binary_op_type.h"
#include "util/arithmetic_op_type/unary_op_type.h"

namespace my_inference::cpu::generic {
    template<typename T>
    struct Traits {
        using VecType = T;

        static T load(const T *p) {
            return *p;
        }

        static T store(T *p, const T value) {
            return *p = value;
        }

        static T add(const T a, const T b) {
            return a + b;
        }

        static T sub(const T a, const T b) {
            return a - b;
        }

        static T mul(const T a, const T b) {
            return a * b;
        }

        static T div(const T a, const T b) {
            return a / b;
        }

        static T max(const T a, const T b) {
            if constexpr (std::is_same_v<T, float>) {
                return fmaxf(a, b);
            }
            return std::max(a, b);
        }

        static T min(const T x, const T y) {
            if constexpr (std::is_same_v<T, float>) {
                return fminf(x, y);
            } else {
                return std::min(x, y);
            }
        }

        static T relu6(const T x) {
            return x < 0 ? 0 : x > 6 ? 6 : x;
        }

        template<UnaryOpType op_type>
        static T unaryOp(T x) {
            if constexpr (op_type == UnaryOpType::Relu6) {
                return Traits::relu6(x);
            }
            return 0;
        }

        template<BinaryOpType op_type>
        static T binaryOp(const T a, const T b) {
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
