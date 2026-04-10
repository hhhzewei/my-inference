//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_GEMM_H
#define MY_INFERENCE_GEMM_H
#include <cstdint>

namespace my_inference::cpu::generic::primitive {
    template<typename T, bool TransA, bool TransB>
    void gemm(const T *a, const T *b,
              const T *bias, const int64_t bias_stride_0, const int64_t bias_stride_1,
              T *y,
              const int64_t M, const int64_t K, const int64_t N,
              const float alpha, const float beta) {
        int64_t strideA[2] = {K, 1};
        if (TransA) {
            strideA[0] = 1;
            strideA[1] = M;
        }
        int64_t strideB[2] = {N, 1};
        if (TransB) {
            strideB[0] = 1;
            strideB[1] = K;
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T ret = 0;
                for (int k = 0; k < K; ++k) {
                    ret += a[i * strideA[0] + k * strideA[1]] * b[k * strideB[0] + j * strideB[1]];
                }
                y[i * N + j] = alpha * ret + beta * bias[i * bias_stride_0 + j * bias_stride_1];
            }
        }
    }
}
#endif //MY_INFERENCE_GEMM_H
