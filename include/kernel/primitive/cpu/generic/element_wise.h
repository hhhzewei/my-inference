//
// Created by hzw on 2026/2/28.
//
#pragma once
#include <cstdint>

#include "backend/isa_traits/cpu_traits.h"
#include "kernel/kernel_args/broadcast_binary_elementwise_args.h"
#include "kernel/kernel_args/elementwise_args.h"
#include "util/arithmetic_op_type/unary_op_type.h"

namespace my_inference::cpu::generic::primitive {
    template<typename T, UnaryOpType op_type>
    void unaryElementWise(
        // 输入数据指针
        const T *a,
        // 输出数据指针 (Output)
        T *b,
        const ElementWiseArgs args) {
        for (int64_t i = 0; i < args.num_elem; ++i) {
            b[i] = Traits<T>::unaryOp < op_type > (a[i]);
        }
    }

    template<typename T, BinaryOpType op_type>
    void binaryElementWise(
        // 输入数据指针
        const T *a, const T *b,
        // 输出数据指针 (Output)
        T *c,
        const ElementWiseArgs args) {
        for (int64_t i = 0; i < args.num_elem; ++i) {
            c[i] = Traits<T>::binaryOp<op_type>(a[i], b[i]);
        }
    }


    template<typename T, BinaryOpType op_type>
    void broadcastBinaryElementWise1D(
        // 输入数据指针
        const T *a, const T *b,
        // 输出数据指针 (Output)
        T *c,
        const BroadcastBinaryElementwiseArgs args) {
        for (int64_t i = 0; i < args.num_elem; ++i) {
            c[i] = Traits<T>::binaryOp < op_type > (a[i * args.a_strides[0]], b[i * args.b_strides[0]]);
        }
    }

    template<typename T, BinaryOpType op_type>
    void broadcastBinaryElementwise2D(
        const T *a,
        const T *b,
        T *c, const BroadcastBinaryElementwiseArgs args) {
        for (int64_t i = 0; i < args.shape[0]; ++i) {
            for (int64_t j = 0; j < args.shape[1]; ++j) {
                c[i * args.shape[1] + j] = Traits<T>::binaryOp < op_type > (
                                               a[i * args.a_strides[0] + j * args.a_strides[1]],
                                               b[i * args.b_strides[0] + j * args.b_strides[1]]);
            }
        }
    }

    template<typename T, BinaryOpType op_type>
    void broadcastBinaryElementWiseND(
        const T *a, const T *b,
        T *c,
        const BroadcastBinaryElementwiseArgs args) {
        int64_t coords[8]{};
        int64_t a_offset = 0, b_offset = 0;
        const int64_t *a_strides = args.a_strides;
        const int64_t *b_strides = args.b_strides;
        for (int64_t c_offset = 0; c_offset < args.num_elem; ++c_offset) {
            c[c_offset] = Traits<T>::binaryOp < op_type > (a[a_offset], b[b_offset]);
            // update coords
            for (int64_t dim_i = args.num_dim - 1; dim_i >= 0; --dim_i) {
                ++coords[dim_i];
                a_offset += a_strides[dim_i];
                b_offset += b_strides[dim_i];
                if (coords[dim_i] < args.shape[dim_i]) {
                    break;
                }
                a_offset -= coords[dim_i] * a_strides[dim_i];
                b_offset -= coords[dim_i] * b_strides[dim_i];
                coords[dim_i] = 0;
            }
        }
    }
}
