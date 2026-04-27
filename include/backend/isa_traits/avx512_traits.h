//
// Created by hzw on 2026/4/24.
//
#pragma once

#include "zmmintrin.h"
#include "util/arithmetic_op_type/binary_op_type.h"
#include "util/arithmetic_op_type/unary_op_type.h"

namespace my_inference::cpu::avx512 {
    template<typename T>
    struct TraitBase {
        constexpr static int64_t VecSize = 64;
        constexpr static int64_t NumPerVec = VecSize / sizeof(T);
    };

    template<typename T>
    struct Traits {
    };

    template<>
    struct Traits<float> : TraitBase<float> {
        using VecType = __m512;
        using MaskType = __mmask16;

        static VecType load(const void *p) {
            return _mm512_load_ps(p);
        }

        static void store(void *p, const VecType vec) {
            _mm512_store_ps(p, vec);
        }

        static VecType setzero() {
            return _mm512_setzero_ps();
        }

        static VecType set1(float v) {
            return _mm512_set1_ps(v);
        }

        static VecType add(VecType a, VecType b) {
            return _mm512_add_ps(a, b);
        }

        static VecType sub(VecType a, VecType b) {
            return _mm512_sub_ps(a, b);
        }

        static VecType mul(VecType a, VecType b) {
            return _mm512_mul_ps(a, b);
        }

        static VecType div(VecType a, VecType b) {
            return _mm512_div_ps(a, b);
        }

        static VecType fmadd(VecType a, VecType b, VecType c) {
            return _mm512_fmadd_ps(a, b, c);
        }

        static float reduce_add(VecType x) {
            return _mm512_reduce_add_ps(x);
        }

        static VecType mask_blend(const MaskType mask, const VecType false_vec, const VecType true_vec) {
            return _mm512_mask_blend_ps(mask, false_vec, true_vec);
        }

        static VecType relu6(VecType x) {
            auto tmp = _mm512_setzero_ps();
            const auto y = _mm512_max_ps(x, tmp);
            tmp = _mm512_set1_ps(6.0f);
            return _mm512_min_ps(y, tmp);
        }

        template<UnaryOpType op_type>
        static VecType unaryOp(VecType x) {
            if constexpr (op_type == UnaryOpType::Relu6) {
                return relu6(x);
            }
            return _mm512_setzero_ps();
        }

        template<BinaryOpType op_type>
        static VecType binaryOp(VecType x1, VecType x2) {
            if constexpr (op_type == BinaryOpType::Add) {
                return add(x1, x2);
            }
            if constexpr (op_type == BinaryOpType::Sub) {
                return sub(x1, x2);
            }

            if constexpr (op_type == BinaryOpType::Mul) {
                return mul(x1, x2);
            }
            if constexpr (op_type == BinaryOpType::Div) {
                return div(x1, x2);
            }
            return _mm512_setzero_ps();
        }
    };
}
